# PlantDoc: Real-time Plant Disease Detection & Advice

**Category:** Deep Learning  
**Tech Stack:** Python, PyTorch, torchvision, transformers, FastAPI, OpenCV, NumPy  

---  

## Table of Contents
- [Project Overview](#project-overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Running the Application](#running-the-application)  
- [API Reference](#api-reference)  
- [Training a New Model](#training-a-new-model)  
- [Inference Demo](#inference-demo)  
- [Testing](#testing)  
- [Docker Support](#docker-support)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  

---  

## Project Overview
PlantDoc is an end‑to‑end solution that detects plant diseases from leaf images in real time and provides actionable advice to growers. The system combines a convolutional backbone (ResNet‑50) with a lightweight transformer head for fine‑grained classification, exposing the model via a FastAPI service. Images captured from a webcam, mobile device, or uploaded file are processed with OpenCV, normalized with NumPy, and classified with a PyTorch model trained on the PlantVillage dataset and custom field samples.

---  

## Features
- **Real‑time detection** using a webcam or video stream.  
- **RESTful API** for remote inference (`/predict`).  
- **Explainability** via Grad‑CAM heatmaps.  
- **Advice engine** that maps disease predictions to treatment recommendations stored in a JSON knowledge base.  
- **GPU acceleration** (CUDA) with automatic fallback to CPU.  
- **Dockerized deployment** for reproducible production environments.  

---  

## Architecture
```
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── routes.py          # API route definitions
│   ├── model.py           # Model loading & inference utilities
│   ├── utils.py           # Image preprocessing, Grad‑CAM, advice lookup
│   └── config.yaml        # Runtime configuration
├── data/
│   ├── raw/               # Original images
│   └── processed/         # Pre‑processed tensors
├── models/
│   ├── plantdoc_resnet_transformer.pt   # Trained checkpoint
│   └── label_encoder.json               # Class‑index mapping
├── notebooks/             # Exploration & training notebooks
├── tests/                 # PyTest suite
└── Dockerfile
```

---  

## Installation

### Prerequisites
- Python ≥ 3.9  
- CUDA‑enabled GPU (optional but recommended)  
- `git`  

### Step‑by‑step

```bash
# Clone the repository
git clone https://github.com/yourusername/plantdoc.git
cd plantdoc

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt`** (excerpt):
```
fastapi==0.110.0
uvicorn[standard]==0.27.0
torch==2.2.0
torchvision==0.17.0
transformers==4.40.0
opencv-python==4.9.0.80
numpy==1.26.4
pydantic==2.7.0
pyyaml==6.0.1
grad-cam==1.4.6
pytest==8.2.0
```

---  

## Configuration
All runtime options are stored in `app/config.yaml`. Example:

```yaml
model:
  checkpoint_path: "models/plantdoc_resnet_transformer.pt"
  label_map_path: "models/label_encoder.json"
  device: "cuda"   # Set to "cpu" if no GPU is available

api:
  host: "0.0.0.0"
  port: 8000
  reload: true

advice:
  knowledge_base: "data/advice.json"
```

You can override any setting via environment variables using the `APP_` prefix (e.g., `APP_MODEL_DEVICE=cpu`).

---  

## Running the Application

### Development mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000/docs` to explore the interactive Swagger UI.

### Production mode (Gunicorn)
```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app.main:app
```

---  

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | `POST` | Accepts an image file (`multipart/form-data`) and returns disease class, confidence, Grad‑CAM heatmap (base64), and treatment advice. |
| `/health` | `GET` | Simple health check returning `{"status":"ok"}`. |
| `/labels` | `GET` | Returns the list of supported disease labels. |

**Example request (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@samples/leaf.jpg" \
  -H "Accept: application/json"
```

**Typical JSON response:**
```json
{
  "label": "Apple_scab",
  "confidence": 0.9623,
  "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "advice": "Apply copper-based fungicide weekly for 3 weeks."
}
```

---  

## Training a New Model

1. **Prepare data** – place raw images in `data/raw/<disease_name>/`.  
2. **Run the preprocessing notebook** (`notebooks/01_preprocess.ipynb`) to create `torchvision` datasets saved under `data/processed/`.  
3. **Start training**:

```bash
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    --data-dir data/processed \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --output models/plantdoc_resnet_transformer.pt \
    --label-map models/label_encoder.json
```

Training script (`train.py`) logs metrics to `logs/` and saves the best checkpoint based on validation accuracy.

---  

## Inference Demo

A minimal CLI helper is provided for quick testing:

```bash
python scripts/infer.py \
    --image samples/leaf.jpg \
    --checkpoint models/plantdoc_resnet_transformer.pt \
    --label-map models/label_encoder.json
```

The script prints the predicted label, confidence, and saves a side‑by‑side image (`output.png`) showing the original leaf and the Grad‑CAM overlay.

---  

## Testing

Run the full test suite with:

```bash
pytest -vv
```

Key test modules:

- `tests/test_model.py` – verifies checkpoint loading and deterministic output for a fixed seed.  
- `tests/test_api.py` – uses FastAPI’s `TestClient` to ensure `/predict` returns a 200 status and proper JSON schema.  
- `tests/test_utils.py` – checks image preprocessing, Grad‑CAM generation, and advice lookup.

All tests are expected to pass on both CPU and GPU environments.

---  

## Docker Support

### Build the image
```bash
docker build -t plantdoc:latest .
```

### Run the container
```bash
docker run -d -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    plantdoc:latest
```

The container automatically detects CUDA support if the host has the NVIDIA runtime configured.

---  

## Contributing

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feat/your-feature`).  
3. Write tests for any new functionality.  
4. Ensure `black`, `isort`, and `flake8` pass (`make lint`).  
5. Submit a pull request with a clear description of the changes.

Please adhere to the **PEP 8** style guide and keep the documentation up to date.

---  

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---  

## Acknowledgements
- The PlantVillage dataset contributors.  
- PyTorch and the Hugging Face Transformers community.  
- FastAPI developers for the excellent asynchronous framework.  

For any questions or issues, please open an issue on GitHub or contact `contact@plantdoc.ai`.