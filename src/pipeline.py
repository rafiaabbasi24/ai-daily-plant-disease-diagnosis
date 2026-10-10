import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

# Transformers for generating textual advice (optional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

# FastAPI for optional endpoint exposure
try:
    from fastapi import FastAPI, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    UploadFile = None
    JSONResponse = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _load_image(file_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from a file path and convert it to RGB PIL.Image.

    Args:
        file_path: Path to the image file.

    Returns:
        A PIL.Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be opened as an image.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    img = Image.open(file_path).convert("RGB")
    logger.debug("Loaded image %s (size=%s)", file_path, img.size)
    return img


def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV BGR image to a PIL RGB image.

    Args:
        image: NumPy array in BGR format.

    Returns:
        A PIL.Image in RGB mode.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    logger.debug("Converted CV2 image to PIL (size=%s)", pil_img.size)
    return pil_img


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #


class PlantDiseasePipeline:
    """
    End‑to‑end pipeline for preprocessing plant images, running a classification
    model, and optionally generating textual advice.

    The class loads a PyTorch image classification model (e.g., a ResNet) and,
    if available, a causal language model for advice generation.

    Example:
        >>> pipeline = PlantDiseasePipeline(
        ...     model_path="models/plant_disease_resnet.pt",
        ...     label_map_path="models/label_map.txt",
        ...     advice_model_name="gpt2"
        ... )
        >>> preds = pipeline.predict_from_path("sample.jpg")
        >>> print(preds)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        label_map_path: Union[str, Path],
        device: str = "auto",
        advice_model_name: str = "",
        advice_cache_dir: Union[str, Path] = "cache/advice",
        transform: nn.Module = None,
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            model_path: Path to a TorchScript or ``.pt`` state‑dict file.
            label_map_path: Path to a text file mapping class indices to names.
                Expected format: one label per line, ordered by class index.
            device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (default). ``"auto"``
                selects CUDA if available.
            advice_model_name: Name of a HuggingFace causal LM (e.g. ``"gpt2"``).
                If empty, advice generation is disabled.
            advice_cache_dir: Directory where the transformer model will be cached.
            transform: Optional torchvision transform. If ``None`` a default
                transform suitable for most classification models is used.
        """
        self.device = self._resolve_device(device)
        self.model = self._load_image_model(model_path, self.device)
        self.label_map = self._load_label_map(label_map_path)
        self.transform = transform or self._default_transform()
        self.advice_model, self.advice_tokenizer = self._load_advice_model(
            advice_model_name, advice_cache_dir
        )
        logger.info("PlantDiseasePipeline initialised on %s", self.device)

    # ------------------------------------------------------------------- #
    # Device handling
    # ------------------------------------------------------------------- #

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        """Resolve the device string to a ``torch.device``."""
        if requested == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested)
        logger.debug("Resolved device: %s", device)
        return device

    # ------------------------------------------------------------------- #
    # Model loading
    # ------------------------------------------------------------------- #

    @staticmethod
    def _load_image_model(model_path: Union[str, Path], device: torch.device) -> nn.Module:
        """
        Load a classification model.

        The function tries to load a TorchScript model first; if that fails,
        it falls back to loading a state‑dict into a generic ``torchvision.models.resnet18``.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # TorchScript (scripted or traced) model
            model = torch.jit.load(str(model_path), map_location=device)
            logger.info("Loaded TorchScript model from %s", model_path)
        except RuntimeError:
            # Assume a state dict for a ResNet‑18 architecture
            from torchvision.models import resnet18

            model = resnet18(pretrained=False)
            state_dict = torch.load(str(model_path), map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Loaded state‑dict into ResNet‑18 from %s", model_path)

        model.eval()
        model.to(device)
        return model

    @staticmethod
    def _load_label_map(label_map_path: Union[str, Path]) -> List[str]:
        """
        Load a simple newline‑separated label map.

        Returns:
            List where index corresponds to class id.
        """
        label_map_path = Path(label_map_path)
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map file not found: {label_map_path}")

        with label_map_path.open("r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if not labels:
            raise ValueError("Label map file is empty")
        logger.debug("Loaded %d class labels", len(labels))
        return labels

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """
        Return a torchvision transform that:
        - Resizes the shorter side to 256,
        - Center‑crops to 224,
        - Converts to tensor,
        - Normalises with ImageNet statistics.
        """
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_advice_model(
        model_name: str, cache_dir: Union[str, Path]
    ) -> Tuple[Any, Any]:
        """
        Load a HuggingFace causal language model and tokenizer for advice generation.

        If ``model_name`` is falsy or the ``transformers`` library is unavailable,
        returns ``(None, None)``.

        Args:
            model_name: Name of the model on the HF hub.
            cache_dir: Directory to cache the model.

        Returns:
            Tuple of (model, tokenizer) or (None, None).
        """
        if not model_name:
            logger.info("Advice generation disabled (no model name provided).")
            return None, None

        if AutoModelForCausalLM is None or AutoTokenizer is None:
            logger.warning(
                "Transformers library not installed; advice generation will be disabled."
            )
            return None, None

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir))
        model.eval()
        logger.info("Loaded advice model '%s' with cache dir %s", model_name, cache_dir)
        return model, tokenizer

    # ------------------------------------------------------------------- #
    # Pre‑processing
    # ------------------------------------------------------------------- #

    def preprocess(self, image: Union[np.ndarray, Image.Image, str, Path]) -> torch.Tensor:
        """
        Convert raw input into a model‑ready tensor.

        Supported input types:
        - ``np.ndarray`` (OpenCV BGR)
        - ``PIL.Image.Image``
        - Path to an image file

        Returns:
            Tensor of shape ``(1, C, H, W)`` on the pipeline's device.
        """
        if isinstance(image, (str, Path)):
            pil_img = _load_image(image)
        elif isinstance(image, np.ndarray):
            pil_img = _cv2_to_pil(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(
                "Unsupported image type. Expected np.ndarray, PIL.Image, or path string."
            )

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logger.debug("Preprocessed image to tensor shape %s", tensor.shape)
        return tensor

    # ------------------------------------------------------------------- #
    # Inference
    # ------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Run forward pass and return top‑k predictions.

        Args:
            image_tensor: Tensor produced by :meth:`preprocess`.
            top_k: Number of top predictions to return.

        Returns:
            List of dictionaries with ``label``, ``score`` and optionally ``advice``.
        """
        if image_tensor.ndim != 4:
            raise ValueError("Expected image tensor of shape (1, C, H, W)")

        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        top_scores, top_idxs = torch.topk(probs, k=top_k)
        results = []
        for score, idx in zip(top_scores.tolist(), top_idxs.tolist()):
            label = self.label_map[idx] if idx < len(self.label_map) else f"class_{idx}"
            entry: Dict[str, Any] = {"label": label, "score": score}
            if self.advice_model and self.advice_tokenizer:
                entry["advice"] = self._generate_advice(label)
            results.append(entry)

        logger.info("Prediction results: %s", results)
        return results

    # ------------------------------------------------------------------- #
    # Advice generation
    # ------------------------------------------------------------------- #

    @torch.no_grad()
    def _generate_advice(self, disease_label: str, max_new_tokens: int = 80) -> str:
        """
        Generate short textual advice for a given disease label.

        The prompt is kept deterministic to avoid hallucinations; the model is
        prompted with a concise instruction.

        Args:
            disease_label: The name of the detected disease.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated advice string (stripped of leading/trailing whitespace).
        """
        if not self.advice_model or not self.advice_tokenizer:
            return ""

        prompt = f"The plant is affected by {disease_label}. Provide a concise care tip."
        inputs = self.advice_tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output_ids = self.advice_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.advice_tokenizer.eos_token_id,
        )
        # Remove the prompt part
        generated = output_ids[0][inputs.shape[-1] :]
        advice = self.advice_tokenizer.decode(generated, skip_special_tokens=True).strip()
        logger.debug("Generated advice for '%s': %s", disease_label, advice)
        return advice

    # ------------------------------------------------------------------- #
    # Convenience wrappers
    # ------------------------------------------------------------------- #

    def predict_from_path(self, image_path: Union[str, Path], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Load an image from disk, preprocess, and predict.

        Args:
            image_path: Path to the image file.
            top_k: Number of top predictions to return.

        Returns:
            Same format as :meth:`predict`.
        """
        tensor = self.preprocess(image_path)
        return self.predict(tensor, top_k=top_k)

    def predict_from_numpy(self, image_array: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict from an OpenCV BGR ``np.ndarray``.

        Args:
            image_array: BGR image array.
            top_k: Number of top predictions to return.

        Returns:
            Same format as :meth:`predict`.
        """
        tensor = self.preprocess(image_array)
        return self.predict(tensor, top_k=top_k)

    # ------------------------------------------------------------------- #
    # Optional FastAPI integration
    # ------------------------------------------------------------------- #

    @staticmethod
    def create_fastapi_app(pipeline: "PlantDiseasePipeline") -> Any:
        """
        Build a minimal FastAPI app exposing a single ``/predict`` endpoint.

        The endpoint accepts multipart/form‑data with a file field named ``image``.
        It returns JSON with the top‑k predictions.

        Returns:
            An instance of ``FastAPI`` (or ``None`` if FastAPI is not installed).
        """
        if FastAPI is None:
            logger.error("FastAPI is not installed; cannot create API app.")
            return None

        app = FastAPI(title="Plant Disease Detection API")

        @app.post("/predict")
        async def predict_endpoint(
            image: UploadFile,
            top_k: int = 3,
        ) -> JSONResponse:
            """
            Predict plant disease from an uploaded image.

            Parameters
            ----------
            image: Uploaded image file (any format supported by OpenCV).
            top_k: Number of predictions to return (default 3).

            Returns
            -------
            JSONResponse with ``predictions`` key.
            """
            try:
                # Read file into memory
                content = await image.read()
                np_arr = np.frombuffer(content, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_img is None:
                    raise ValueError("Unable to decode image")
                results = pipeline.predict_from_numpy(cv_img, top_k=top_k)
                return JSONResponse(content={"predictions": results})
            except Exception as exc:
                logger.exception("Prediction failed: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))

        logger.info("FastAPI app created with /predict endpoint")
        return app

# --------------------------------------------------------------------------- #
# Unit‑test friendly utilities (executed only when run as a script)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Simple sanity check – this block runs only when the module is executed directly.
    import argparse

    parser = argparse.ArgumentParser(description="Run a quick inference test.")
    parser.add_argument("image_path", type=str, help="Path to an image file.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/plant_disease_resnet.pt",
        help="Path to the classification model.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="models/label_map.txt",
        help="Path to the label map file.",
    )
    parser.add_argument(
        "--advice-model",
        type=str,
        default="gpt2",
        help="HuggingFace model name for advice generation (empty to disable).",
    )
    args = parser.parse_args()

    pipeline = PlantDiseasePipeline(
        model_path=args.model,
        label_map_path=args.labels,
        advice_model_name=args.advice_model,
    )
    predictions = pipeline.predict_from_path(args.image_path, top_k=3)
    print("Top predictions:")
    for pred in predictions:
        print(f"- {pred['label']}: {pred['score']:.4f}")
        if "advice" in pred:
            print(f"  Advice: {pred['advice']}")