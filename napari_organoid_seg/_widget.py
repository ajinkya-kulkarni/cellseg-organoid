"""Minimal napari widget for organoid instance segmentation."""

import math
import os
from pathlib import Path

import numpy as np
from napari import current_viewer
from napari.layers import Image
from napari.qt import thread_worker
from napari.utils.notifications import show_warning

from config import PATCH_SIZE, TILE_OVERLAP

_MODEL_ENV = "ORGANOID_SEG_MODEL"
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_MAX_DIM = 2000

_predictor = None
_predictor_path: Path | None = None
_active_worker = None


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


def _tile_count(spatial_shape: tuple[int, int]) -> int:
    """Return the number of tiles used by the regular halo-stitching grid."""
    stride = PATCH_SIZE - TILE_OVERLAP

    def count(dim_size: int) -> int:
        if dim_size <= PATCH_SIZE:
            return 1
        return math.ceil((dim_size - PATCH_SIZE) / stride) + 1

    height, width = spatial_shape
    return count(height) * count(width)


@thread_worker
def _segment_worker(image_data: np.ndarray):
    """Run tiled inference off the Qt main thread and yield progress units."""
    from large_inference import iter_large_image

    predictor = _get_predictor()
    labels = yield from iter_large_image(
        predictor,
        image_data,
        overlap=TILE_OVERLAP,
        batch_size=_DEFAULT_BATCH_SIZE,
    )
    return labels


def _clear_worker(worker) -> None:
    global _active_worker
    if _active_worker is worker:
        _active_worker = None


def segment_organoids(image: Image) -> None:
    """Start background segmentation and add the final editable Labels layer."""
    global _active_worker

    if _active_worker is not None:
        show_warning("Organoid segmentation is already running.")
        return

    # Copy before starting the worker so the background thread sees a stable array
    # even if the user continues interacting with the source layer.
    data = _validate_image(image).copy()
    layer_name = f"{image.name} - organoids"
    scale = tuple(image.scale)
    translate = tuple(image.translate)

    # One work unit per tile plus one final unit for global StarDist NMS/polygons.
    total_work = _tile_count(data.shape[:2]) + 1

    worker = _segment_worker(
        data,
        _progress={
            "total": total_work,
            "desc": "Segmenting organoids",
        },
    )
    _active_worker = worker

    def add_result(labels: np.ndarray) -> None:
        viewer = current_viewer()
        if viewer is None:
            return
        viewer.add_labels(
            labels,
            name=layer_name,
            scale=scale,
            translate=translate,
        )

    worker.returned.connect(add_result)
    worker.finished.connect(lambda: _clear_worker(worker))
    worker.start()
