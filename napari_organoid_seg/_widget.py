"""Minimal napari widget for organoid instance segmentation."""

import os
from pathlib import Path

import numpy as np
from napari.layers import Image
from napari.types import LayerDataTuple

from config import TILE_OVERLAP

_MODEL_ENV = "ORGANOID_SEG_MODEL"
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_MAX_DIM = 2000

_predictor = None
_predictor_path: Path | None = None


def _resolve_model_path() -> Path:
    """Find the ONNX model for local development or packaged use."""
    candidates: list[Path] = []

    env_path = os.environ.get(_MODEL_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    package_model = Path(__file__).parent / "models" / "best.onnx"
    candidates.append(package_model)

    repo_model = Path(__file__).resolve().parent.parent / "models" / "best.onnx"
    candidates.append(repo_model)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the organoid ONNX model. "
        f"Set {_MODEL_ENV} to best.onnx or place it in models/best.onnx. "
        f"Searched:\n{searched}"
    )


def _get_predictor():
    """Create the ONNX Runtime session lazily and reuse it."""
    global _predictor, _predictor_path

    from onnx_inference import OrganoidONNXPredictor

    model_path = _resolve_model_path()
    if _predictor is None or _predictor_path != model_path:
        _predictor = OrganoidONNXPredictor(model_path)
        _predictor_path = model_path

    return _predictor


def _validate_image(layer: Image) -> np.ndarray:
    """Return supported 2D grayscale/RGB image data as a NumPy array."""
    data = np.asarray(layer.data)

    if data.ndim == 2:
        spatial_shape = data.shape
    elif data.ndim == 3 and bool(layer.rgb) and data.shape[-1] in (3, 4):
        spatial_shape = data.shape[:2]
    else:
        raise ValueError(
            "Organoid segmentation currently supports one 2D grayscale or RGB image. "
            f"Got data shape {data.shape}."
        )

    if max(spatial_shape) > _DEFAULT_MAX_DIM:
        raise ValueError(
            f"Image is {spatial_shape[1]}x{spatial_shape[0]}. "
            f"The current napari MVP limits each side to {_DEFAULT_MAX_DIM}px "
            "to avoid excessive StarDist ray-map memory usage."
        )

    return data


def segment_organoids(image: Image) -> LayerDataTuple:
    """Segment one napari Image layer and return an editable Labels layer."""
    from large_inference import predict_large_image

    data = _validate_image(image)
    predictor = _get_predictor()

    labels = predict_large_image(
        predictor,
        data,
        overlap=TILE_OVERLAP,
        batch_size=_DEFAULT_BATCH_SIZE,
    )

    metadata = {
        "name": f"{image.name} - organoids",
        "scale": tuple(image.scale),
        "translate": tuple(image.translate),
    }
    return labels, metadata, "labels"
