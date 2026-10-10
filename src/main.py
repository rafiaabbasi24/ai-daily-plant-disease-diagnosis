import io
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent.parent / "model" / "plant_disease_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("plantdoc")

# ----------------------------------------------------------------------
# FastAPI app initialization
# ----------------------------------------------------------------------
app = FastAPI(
    title="PlantDoc: Real-time Plant Disease Detection & Advice",
    description=(
        "Upload an image of a plant leaf and receive a disease prediction "
        "with confidence scores and treatment advice."
    ),
    version="1.0.0",
)

# Enable CORS for all origins (adjust in production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Model and utilities
# ----------------------------------------------------------------------
class PredictionResult(BaseModel):
    disease: str
    confidence: float
    advice: str


def load_model(model_path: Path) -> torch.nn.Module:
    """
    Load a TorchScript or state‑dict model from ``model_path`` onto ``DEVICE``.
    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model cannot be loaded.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        # Attempt to load a TorchScript model first
        model = torch.jit.load(str(model_path), map_location=DEVICE)
        logger.info("Loaded TorchScript model.")
    except RuntimeError:
        # Fallback to a regular state‑dict model
        model = torch.nn.Module()  # placeholder for type checking
        state_dict = torch.load(str(model_path), map_location=DEVICE)
        # The actual architecture must be defined elsewhere; for demo we assume
        # a generic torchvision ResNet18 with modified final layer.
        from torchvision import models

        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(DISEASE_MAP))
        model.load_state_dict(state_dict)
        logger.info("Loaded state‑dict model.")
    model.eval()
    model.to(DEVICE)
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes to a normalized torch tensor suitable for the model.
    The image is resized to 224×224, normalized with ImageNet stats, and batched.
    """
    # Decode image bytes to a NumPy array (BGR format)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Unable to decode image. Ensure it is a valid image file.")

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to 224×224 (common for many CNNs)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0

    # ImageNet mean/std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    # HWC → CHW and add batch dimension
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def get_advice(disease: str) -> str:
    """
    Return a short advice string for the given disease.
    In a real project this could query a database or external API.
    """
    advice_map = {
        "Apple Scab": "Apply appropriate fungicide and remove infected leaves.",
        "Apple Black Rot": "Prune affected branches and use copper‑based sprays.",
        "Tomato Early Blight": "Rotate crops and apply chlorothalonil.",
        "Healthy": "No disease detected. Continue regular care.",
    }
    return advice_map.get(disease, "Consult an agronomist for detailed guidance.")


# ----------------------------------------------------------------------
# Disease label mapping (index → human readable name)
# ----------------------------------------------------------------------
DISEASE_MAP: Dict[int, str] = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Tomato Early Blight",
    3: "Healthy",
    # Extend with additional classes as needed.
}

# Load model at startup
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as exc:
    logger.exception("Failed to load model during startup.")
    model = None  # The endpoint will raise if model is None


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """
    Simple health‑check endpoint used by orchestration tools.
    Returns a JSON payload confirming the service is up.
    """
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Prediction"],
    summary="Predict plant disease from an uploaded leaf image",
)
async def predict(file: UploadFile = File(...)) -> PredictionResult:
    """
    Accepts an image file, runs the disease classification model,
    and returns the most likely disease together with confidence and advice.

    Args:
        file: Uploaded image (must be JPEG/PNG).

    Returns:
        PredictionResult containing disease name, confidence (0‑1), and advice.
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed types: {ALLOWED_EXTENSIONS}",
        )

    # Ensure model is ready
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Try again later.",
        )

    # Read file contents
    try:
        contents = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file.")
        raise HTTPException(status_code=400, detail="Could not read uploaded file.") from exc

    # Preprocess and predict
    try:
        input_tensor = preprocess_image(contents)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    except Exception as exc:
        logger.exception("Error during model inference.")
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    # Determine top prediction
    top_idx = int(np.argmax(probabilities))
    disease = DISEASE_MAP.get(top_idx, "Unknown")
    confidence = float(probabilities[top_idx])
    advice = get_advice(disease)

    logger.info(
        "Prediction: %s (confidence %.2f) for file %s",
        disease,
        confidence,
        file.filename,
    )

    return PredictionResult(disease=disease, confidence=confidence, advice=advice)


# ----------------------------------------------------------------------
# Run with: python -m src.main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )