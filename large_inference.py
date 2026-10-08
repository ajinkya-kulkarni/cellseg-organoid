"""Large-image ONNX inference by stitching dense StarDist outputs.

Pipeline:
  RGB image
    -> overlapping 256x256 tiles
    -> batched ONNX inference
    -> weighted blending of raw binary logits + ray maps
    -> sigmoid on the stitched logits
    -> one global StarDist postprocess

This avoids per-tile instance-ID merging.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from onnx_inference import (
    NMS_IOU_THRESH,
    PATCH_SIZE,
    SCORE_THRESH,
    OrganoidONNXPredictor,
    sigmoid,
)
from stardist_postproc import post_proc_stardist

OVERLAP = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_DIM = 2000


def patch_starts(dim_size: int, patch_size: int, overlap: int) -> list[int]:
    """Return tile starts that cover the full dimension."""
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    if dim_size <= patch_size:
        return [0]

    stride = patch_size - overlap
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] + patch_size < dim_size:
        starts.append(dim_size - patch_size)
    return starts


def _blend_window(size: int = PATCH_SIZE, edge_weight: float = 0.05) -> np.ndarray:
    """Raised-cosine 2D tile weight with non-zero image-edge support."""
    if not 0.0 < edge_weight <= 1.0:
        raise ValueError("edge_weight must be in (0, 1]")

    one_d = np.hanning(size).astype(np.float32)
    one_d = edge_weight + (1.0 - edge_weight) * one_d
    window = np.outer(one_d, one_d).astype(np.float32)
    return window


def _extract_tile(
    image_rgb: np.ndarray,
    y: int,
    x: int,
    patch_size: int,
) -> tuple[np.ndarray, int, int]:
    """Extract one tile and pad bottom/right when the image is smaller than a tile."""
    height, width = image_rgb.shape[:2]
    valid_h = min(patch_size, height - y)
    valid_w = min(patch_size, width - x)

    tile = image_rgb[y : y + valid_h, x : x + valid_w]

    if valid_h == patch_size and valid_w == patch_size:
        return tile, valid_h, valid_w

    pad_bottom = patch_size - valid_h
    pad_right = patch_size - valid_w

    tile = cv2.copyMakeBorder(
        tile,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return tile, valid_h, valid_w


def predict_large_dense(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stitched raw logits (H,W) and ray maps (32,H,W)."""
    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim == 2:
        image_rgb = np.repeat(image_rgb[..., None], 3, axis=2)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 1:
        image_rgb = np.repeat(image_rgb, 3, axis=2)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[..., :3]

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image_rgb.shape}")
    if patch_size != PATCH_SIZE:
        raise ValueError(
            f"This ONNX model has fixed {PATCH_SIZE}x{PATCH_SIZE} spatial input; "
            f"got patch_size={patch_size}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    height, width = image_rgb.shape[:2]
    x_starts = patch_starts(width, patch_size, overlap)
    y_starts = patch_starts(height, patch_size, overlap)

    weights = _blend_window(patch_size)
    logit_sum = np.zeros((height, width), dtype=np.float32)
    ray_sum = np.zeros((32, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    tiles: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int]] = []

    def flush() -> None:
        nonlocal tiles, metadata
        if not tiles:
            return

        logits_batch, rays_batch = predictor.predict_batch_logits(tiles)

        for idx, (y, x, valid_h, valid_w) in enumerate(metadata):
            weight = weights[:valid_h, :valid_w]
            y_slice = slice(y, y + valid_h)
            x_slice = slice(x, x + valid_w)

            logit_sum[y_slice, x_slice] += (
                logits_batch[idx, :valid_h, :valid_w] * weight
            )
            ray_sum[:, y_slice, x_slice] += (
                rays_batch[idx, :, :valid_h, :valid_w] * weight[None]
            )
            weight_sum[y_slice, x_slice] += weight

        tiles = []
        metadata = []

    for y in y_starts:
        for x in x_starts:
            tile, valid_h, valid_w = _extract_tile(
                image_rgb,
                y,
                x,
                patch_size,
            )
            tiles.append(tile)
            metadata.append((y, x, valid_h, valid_w))

            if len(tiles) >= batch_size:
                flush()

    flush()

    if np.any(weight_sum <= 0):
        raise RuntimeError("Dense stitching left uncovered pixels")

    logits = logit_sum / weight_sum
    rays = ray_sum / weight_sum[None]

    return logits.astype(np.float32, copy=False), rays.astype(np.float32, copy=False)


def predict_large_image(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    score_thresh: float = SCORE_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
) -> np.ndarray:
    """Return a full-resolution int32 instance-label mask."""
    logits, rays = predict_large_dense(
        predictor,
        image_rgb,
        patch_size=PATCH_SIZE,
        overlap=overlap,
        batch_size=batch_size,
    )

    probability = sigmoid(logits)
    labels = post_proc_stardist(
        probability,
        rays,
        score_thresh=score_thresh,
        iou_thresh=nms_iou_thresh,
        trim_bboxes=True,
    )
    return labels.astype(np.int32, copy=False)


def _save_preview(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save a lightweight RGB overlay preview for sanity checking."""
    preview = image_rgb.copy()
    if preview.dtype != np.uint8:
        preview = cv2.normalize(
            preview.astype(np.float32),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    boundaries = np.zeros(labels.shape, dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = (labels == label).astype(np.uint8)
        contour_mask = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
        boundaries[contour_mask > 0] = 255

    overlay = preview.copy()
    overlay[boundaries > 0] = (255, 0, 0)
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tiled ONNX StarDist inference on a full RGB image"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/large_labels.npy"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Optional RGB overlay PNG path",
    )
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--score-thresh", type=float, default=SCORE_THRESH)
    parser.add_argument("--nms-iou-thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Safety limit per image side; use 0 to disable (default: 2000)",
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    if image_bgr.ndim == 2:
        image_rgb = np.repeat(image_bgr[..., None], 3, axis=2)
    elif image_bgr.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width = image_rgb.shape[:2]
    if args.max_dim > 0 and max(height, width) > args.max_dim:
        raise ValueError(
            f"Image is {width}x{height}; default safety limit is "
            f"{args.max_dim}px per side. Increase --max-dim explicitly if intended."
        )

    predictor = OrganoidONNXPredictor(args.model)

    labels = predict_large_image(
        predictor,
        image_rgb,
        overlap=args.overlap,
        batch_size=args.batch_size,
        score_thresh=args.score_thresh,
        nms_iou_thresh=args.nms_iou_thresh,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)

    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        _save_preview(image_rgb, labels, args.preview)

    x_tiles = len(patch_starts(width, PATCH_SIZE, args.overlap))
    y_tiles = len(patch_starts(height, PATCH_SIZE, args.overlap))

    print(f"Image:       {width}x{height}")
    print(f"Tiles:       {x_tiles} x {y_tiles} = {x_tiles * y_tiles}")
    print(f"Patch:       {PATCH_SIZE}")
    print(f"Overlap:     {args.overlap}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Instances:   {len(np.unique(labels)) - 1}")
    print(f"Labels:      {args.output}")
    if args.preview is not None:
        print(f"Preview:     {args.preview}")


if __name__ == "__main__":
    main()
