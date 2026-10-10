import pathlib
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

# Import the functions to be tested. Adjust the import path according to the actual
# project layout if necessary.
from plantdoc.model import load_model, predict, preprocess_image


def _dummy_linear_model(num_features: int = 10, num_classes: int = 5) -> nn.Module:
    """
    Create a lightweight dummy model for testing.

    The model consists of a single linear layer so that forward passes are fast
    and deterministic.

    Returns:
        nn.Module: The dummy model.
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(num_features, num_classes))
    # Initialise weights to a known constant for reproducibility.
    torch.nn.init.constant_(model[1].weight, 0.1)
    torch.nn.init.constant_(model[1].bias, 0.0)
    return model


def test_load_model_returns_module(tmp_path: pathlib.Path):
    """
    Verify that ``load_model`` returns a ``torch.nn.Module`` instance.

    The test patches ``torch.load`` to return a dummy model, avoiding any
    filesystem I/O or heavyweight weight loading.
    """
    dummy_path = tmp_path / "dummy_model.pth"
    dummy_path.touch()

    dummy_model = _dummy_linear_model()

    with mock.patch("torch.load", return_value=dummy_model) as mock_load:
        model = load_model(str(dummy_path), device="cpu")
        mock_load.assert_called_once_with(str(dummy_path), map_location="cpu")
        assert isinstance(model, nn.Module), "load_model should return a torch.nn.Module"
        # Ensure the returned model is exactly the dummy we patched.
        assert model is dummy_model


def test_preprocess_image_normalization():
    """
    Ensure that ``preprocess_image`` correctly converts a NumPy image to a
    normalized torch tensor suitable for model input.

    The function is expected to:
    - Convert the image to float32.
    - Scale pixel values from [0, 255] to [0, 1].
    - Apply ImageNet mean/std normalization.
    - Return a tensor of shape (C, H, W).
    """
    # Create a dummy image: 224x224 with 3 channels, all pixel values = 128.
    dummy_image = np.full((224, 224, 3), 128, dtype=np.uint8)

    tensor = preprocess_image(dummy_image)

    # Verify type and shape.
    assert isinstance(tensor, torch.Tensor), "Output must be a torch.Tensor"
    assert tensor.shape == (3, 224, 224), "Tensor shape should be (C, H, W)"

    # Verify normalization: after scaling to [0,1] the value is 0.50196...
    # ImageNet mean for each channel is [0.485, 0.456, 0.406].
    # Expected normalized value = (0.50196 - mean) / std.
    scaled = 128 / 255.0
    expected = (scaled - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    # Compare a single pixel from each channel.
    for c in range(3):
        assert torch.isclose(tensor[c, 0, 0], expected[c], atol=1e-5)


def test_predict_returns_expected_shape():
    """
    Test that ``predict`` runs a forward pass and returns a NumPy array with the
    expected shape (num_classes,).

    A dummy model with a known output dimension is used.
    """
    # Dummy model: input size 3*224*224 flattened -> 5 classes.
    dummy_model = _dummy_linear_model(num_features=3 * 224 * 224, num_classes=5)
    dummy_model.eval()

    # Create a dummy image (all zeros) that matches the expected input size.
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Run prediction.
    output = predict(dummy_model, dummy_image)

    # Verify output type and shape.
    assert isinstance(output, np.ndarray), "Predict should return a NumPy array"
    assert output.shape == (5,), "Output shape should match number of classes"

    # Since the model weights are constant (0.1) and input is zero, the logits
    # should be all zeros (bias is zero). Apply softmax in ``predict`` if it
    # exists; otherwise check raw logits.
    # Here we accept either zeros or a uniform distribution.
    if np.allclose(output, 0):
        # Raw logits case.
        assert np.all(output == 0)
    else:
        # Softmax case: uniform distribution.
        assert np.allclose(output, np.full_like(output, 1 / 5), atol=1e-5)


def test_predict_raises_on_invalid_input():
    """
    ``predict`` should raise a ``ValueError`` when the input image does not have
    the expected three channels.
    """
    dummy_model = _dummy_linear_model(num_features=3 * 224 * 224, num_classes=5)
    dummy_model.eval()

    # Create an invalid image with only one channel.
    invalid_image = np.zeros((224, 224), dtype=np.uint8)  # shape (H, W)

    with pytest.raises(ValueError, match="expected a 3-channel image"):
        predict(dummy_model, invalid_image)


def test_predict_handles_device_argument(monkeypatch):
    """
    Verify that ``predict`` respects the device of the model (CPU vs. CUDA).
    The test forces the model to be on CPU and ensures the output is still a
    NumPy array without raising device‑related errors.
    """
    dummy_model = _dummy_linear_model(num_features=3 * 224 * 224, num_classes=3)
    dummy_model.to("cpu")
    dummy_model.eval()

    dummy_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # No GPU is required; ensure the function does not attempt to move tensors
    # to a non‑existent device.
    with monkeypatch.context() as m:
        m.setattr(torch, "cuda", mock.Mock(is_available=lambda: False))
        output = predict(dummy_model, dummy_image)

    assert isinstance(output, np.ndarray)
    assert output.shape == (3,)


# End of file.