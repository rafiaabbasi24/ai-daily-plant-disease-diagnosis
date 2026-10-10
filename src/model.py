import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Configure module level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PlantDiseaseDetector:
    """
    Vision Transformer based plant disease classifier with built‑in advice generation.

    The class wraps a HuggingFace ``ViTForImageClassification`` model and a
    ``ViTFeatureExtractor``.  It provides a simple ``predict`` method that returns
    the predicted disease label, confidence score and a textual advice string.

    Example
    -------
    >>> detector = PlantDiseaseDetector()
    >>> img = cv2.imread("leaf.jpg")
    >>> label, confidence, advice = detector.predict(img)
    >>> print(label, confidence, advice)
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Union[str, torch.device, None] = None,
        class_names: Union[List[str], None] = None,
        advice_dict: Union[Dict[str, str], None] = None,
        cache_dir: Union[str, Path, None] = None,
    ) -> None:
        """
        Initialise the detector.

        Parameters
        ----------
        model_name: str
            Identifier of a pretrained ViT model on the HuggingFace Hub.
        device: str | torch.device | None
            Torch device to run inference on. If ``None`` the function will
            automatically select ``cuda`` if available, otherwise ``cpu``.
        class_names: list[str] | None
            Optional explicit list of class names. If omitted the model's
            configuration is used.
        advice_dict: dict[str, str] | None
            Mapping from disease label to advice text. Missing entries will be
            filled with a generic fallback.
        cache_dir: str | Path | None
            Directory to cache the model files. Uses the default HuggingFace
            cache if ``None``.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Using device: %s", self.device)

        # Load model and feature extractor
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        logger.info("Loaded ViT model '%s'", model_name)

        self.feature_extractor: ViTFeatureExtractor = ViTFeatureExtractor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        logger.info("Loaded feature extractor for model '%s'", model_name)

        # Resolve class names
        if class_names is not None:
            self.class_names = class_names
        else:
            # The model's config stores the label mapping
            self.class_names = self.model.config.id2label.values()
            self.class_names = list(self.class_names)  # type: ignore[assignment]

        # Advice dictionary handling
        self.advice_dict = advice_dict or {}
        self._populate_default_advice()

        # Transform for additional preprocessing (if needed)
        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),  # converts HWC uint8 to CHW float in [0,1]
                transforms.Normalize(
                    mean=self.feature_extractor.image_mean,
                    std=self.feature_extractor.image_std,
                ),
            ]
        )

    def _populate_default_advice(self) -> None:
        """
        Fill missing advice entries with a generic message.
        """
        generic = (
            "Monitor the plant closely and consider consulting an agronomist "
            "or plant pathologist for further diagnosis."
        )
        for label in self.class_names:
            self.advice_dict.setdefault(label, generic)

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV BGR image to a torch tensor ready for the ViT model.

        Parameters
        ----------
        image: np.ndarray
            Input image in BGR format (as returned by ``cv2.imread``).

        Returns
        -------
        torch.Tensor
            Normalised tensor of shape (1, 3, H, W) on the configured device.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a BGR image with 3 channels, got shape {image.shape}"
            )

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the feature extractor's resizing logic
        inputs = self.feature_extractor(images=rgb_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        return pixel_values

    @torch.no_grad()
    def predict(
        self, image: np.ndarray, top_k: int = 1
    ) -> Tuple[str, float, str, List[Tuple[str, float]]]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image: np.ndarray
            BGR image array (e.g., from ``cv2.imread``).
        top_k: int
            Number of top predictions to return (must be >=1).

        Returns
        -------
        label: str
            Predicted disease label (the top-1 class).
        confidence: float
            Softmax probability of the top-1 prediction.
        advice: str
            Advice string associated with the top-1 label.
        top_predictions: list[tuple[str, float]]
            List of ``(label, probability)`` tuples for the ``top_k`` predictions,
            ordered from most to least confident.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        pixel_values = self._prepare_image(image)

        outputs = self.model(pixel_values)
        logits = outputs.logits  # shape (1, num_classes)
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

        # Get top‑k indices
        top_probs, top_idx = torch.topk(probs, k=top_k, sorted=True)

        top_labels = [self.class_names[idx] for idx in top_idx.cpu().tolist()]
        top_probs_float = top_probs.cpu().tolist()

        # Primary prediction
        label = top_labels[0]
        confidence = top_probs_float[0]
        advice = self.advice_dict.get(label, "")

        top_predictions = list(zip(top_labels, top_probs_float))

        logger.debug(
            "Prediction results: %s (%.4f)", label, confidence
        )
        return label, confidence, advice, top_predictions

    def get_advice(self, label: str) -> str:
        """
        Retrieve advice text for a given disease label.

        Parameters
        ----------
        label: str
            Disease label as returned by ``predict``.

        Returns
        -------
        str
            Advice associated with the label. If the label is unknown,
            a generic fallback message is returned.
        """
        advice = self.advice_dict.get(label)
        if advice is None:
            logger.warning("Advice not found for label '%s', using generic fallback.", label)
            advice = (
                "No specific advice available. Consider seeking expert help "
                "to correctly identify and treat the issue."
            )
        return advice


def load_advice_from_json(path: Union[str, Path]) -> Dict[str, str]:
    """
    Load a JSON file that maps disease labels to advice strings.

    The JSON should be a dictionary where keys are the exact class names
    used by the model and values are the advice text.

    Parameters
    ----------
    path: str | Path
        Path to the JSON file.

    Returns
    -------
    dict[str, str]
        Mapping from label to advice.
    """
    import json

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Advice file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Advice JSON must contain an object mapping labels to strings")

    # Ensure all values are strings
    for key, value in data.items():
        if not isinstance(value, str):
            raise ValueError(f"Advice for label '{key}' must be a string")

    logger.info("Loaded advice for %d labels from %s", len(data), file_path)
    return data


__all__ = [
    "PlantDiseaseDetector",
    "load_advice_from_json",
]