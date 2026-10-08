"""Large-image ONNX inference with halo-cropped dense StarDist stitching.

Pipeline:
  RGB image
    -> regular overlapping 256x256 tile grid
    -> reflect-pad only where the grid extends beyond the image
    -> batched ONNX inference
    -> discard unreliable internal tile borders (halo)
    -> copy trusted dense logits + ray maps into one global canvas
    -> sigmoid on the stitched logits
    -> one global StarDist postprocess

This avoids both per-tile instance merging and averaging edge predictions.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from config import NMS_IOU_THRESH, N_RAYS, PATCH_SIZE, SCORE_THRESH, TILE_OVERLAP
from onnx_inference import OrganoidONNXPredictor, sigmoid
from stardist_postproc import post_proc_stardist

OVERLAP = TILE_OVERLAP
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_DIM = 2000


def patch_starts(dim_size: int, patch_size: int, overlap: int) -> list[int]:
    """Return starts on a regular stride grid that fully covers a dimension.

    The final tile may extend beyond the image. It is reflect-padded during
    extraction instead of being shifted backwards, which keeps every internal
    overlap exactly equal to the requested overlap.
    """
    if dim_size <= 0:
        raise ValueError("dim_size must be greater than zero")
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    if dim_size <= patch_size:
        return [0]

    stride = patch_size - overlap
    n_steps = math.ceil((dim_size - patch_size) / stride)
    return [step * stride for step in range(n_steps + 1)]


def padded_dim(dim_size: int, patch_size: int, overlap: int) -> int:
    """Size covered by the regular tile grid before cropping back to the image."""
    starts = patch_starts(dim_size, patch_size, overlap)
    return starts[-1] + patch_size


def _extract_tile(
    image_rgb: np.ndarray,
    y: int,
    x: int,
    patch_size: int,
) -> np.ndarray:
    """Extract one tile and reflect-pad bottom/right to exactly patch_size."""
    height, width = image_rgb.shape[:2]
    valid_h = min(patch_size, height - y)
    valid_w = min(patch_size, width - x)

    if valid_h <= 0 or valid_w <= 0:
        raise RuntimeError(
            f"Tile start ({x}, {y}) lies outside image {width}x{height}"
        )

    tile = image_rgb[y : y + valid_h, x : x + valid_w]

    pad_bottom = patch_size - valid_h
    pad_right = patch_size - valid_w
    if pad_bottom == 0 and pad_right == 0:
        return tile

    return cv2.copyMakeBorder(
        tile,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _trusted_slice(
    tile_index: int,
    n_tiles: int,
    patch_size: int,
    overlap: int,
) -> slice:
    """Return the trusted tile interval after splitting overlap into halos."""
    if n_tiles == 1:
        return slice(0, patch_size)

    crop_before = overlap // 2
    crop_after = overlap - crop_before

    start = 0 if tile_index == 0 else crop_before
    stop = patch_size if tile_index == n_tiles - 1 else patch_size - crop_after
    return slice(start, stop)


def iter_large_dense(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Yield tile progress and return halo-stitched dense StarDist outputs.

    Yields once per completed tile so GUI callers can report progress without
    changing the synchronous inference result.
    """
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

    canvas_h = padded_dim(height, patch_size, overlap)
    canvas_w = padded_dim(width, patch_size, overlap)

    logits = np.empty((canvas_h, canvas_w), dtype=np.float32)
    rays = np.empty((N_RAYS, canvas_h, canvas_w), dtype=np.float32)
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    tiles: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int]] = []
    completed_tiles = 0
    total_tiles = len(x_starts) * len(y_starts)

    def flush() -> int:
        nonlocal tiles, metadata
        if not tiles:
            return 0

        logits_batch, rays_batch = predictor.predict_batch_logits(tiles)
        batch_count = len(metadata)

        for batch_index, (iy, ix, y, x) in enumerate(metadata):
            src_y = _trusted_slice(iy, len(y_starts), patch_size, overlap)
            src_x = _trusted_slice(ix, len(x_starts), patch_size, overlap)

            dst_y = slice(y + src_y.start, y + src_y.stop)
            dst_x = slice(x + src_x.start, x + src_x.stop)

            logits[dst_y, dst_x] = logits_batch[batch_index, src_y, src_x]
            rays[:, dst_y, dst_x] = rays_batch[
                batch_index, :, src_y, src_x
            ]
            coverage[dst_y, dst_x] += 1

        tiles = []
        metadata = []
        return batch_count

    for iy, y in enumerate(y_starts):
        for ix, x in enumerate(x_starts):
            tiles.append(_extract_tile(image_rgb, y, x, patch_size))
            metadata.append((iy, ix, y, x))

            if len(tiles) >= batch_size:
                flushed = flush()
                for _ in range(flushed):
                    completed_tiles += 1
                    yield completed_tiles, total_tiles

    flushed = flush()
    for _ in range(flushed):
        completed_tiles += 1
        yield completed_tiles, total_tiles

    if not np.all(coverage == 1):
        uncovered = int(np.count_nonzero(coverage == 0))
        duplicated = int(np.count_nonzero(coverage > 1))
        raise RuntimeError(
            "Halo stitching coverage error: "
            f"{uncovered} uncovered pixels, {duplicated} duplicated pixels"
        )

    return (
        logits[:height, :width].astype(np.float32, copy=False),
        rays[:, :height, :width].astype(np.float32, copy=False),
    )


def _consume_generator(generator):
    """Run a progress-yielding generator synchronously and return its result."""
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def predict_large_dense(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return halo-stitched raw logits (H,W) and ray maps (32,H,W)."""
    return _consume_generator(
        iter_large_dense(
            predictor,
            image_rgb,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
        )
    )


def iter_large_image(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    score_thresh: float = SCORE_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
):
    """Yield tile progress, then return the full-resolution label mask."""
    logits, rays = yield from iter_large_dense(
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

    # One final work unit represents global StarDist reconstruction/NMS.
    yield None
    return labels.astype(np.int32, copy=False)


def predict_large_image(
    predictor: OrganoidONNXPredictor,
    image_rgb: np.ndarray,
    overlap: int = OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    score_thresh: float = SCORE_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
) -> np.ndarray:
    """Return a full-resolution int32 instance-label mask."""
    return _consume_generator(
        iter_large_image(
            predictor,
            image_rgb,
            overlap=overlap,
            batch_size=batch_size,
            score_thresh=score_thresh,
            nms_iou_thresh=nms_iou_thresh,
        )
    )


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
        description="Run halo-tiled ONNX StarDist inference on a full RGB image"
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

    x_starts = patch_starts(width, PATCH_SIZE, args.overlap)
    y_starts = patch_starts(height, PATCH_SIZE, args.overlap)

    print(f"Image:       {width}x{height}")
    print(f"Tiles:       {len(x_starts)} x {len(y_starts)} = {len(x_starts) * len(y_starts)}")
    print(f"Patch:       {PATCH_SIZE}")
    print(f"Overlap:     {args.overlap}")
    print(f"Halo:        {args.overlap // 2} / {args.overlap - args.overlap // 2}")
    print(
        "Padded grid: "
        f"{padded_dim(width, PATCH_SIZE, args.overlap)}x"
        f"{padded_dim(height, PATCH_SIZE, args.overlap)}"
    )
    print(f"Batch size:  {args.batch_size}")
    print(f"Instances:   {len(np.unique(labels)) - 1}")
    print(f"Labels:      {args.output}")
    if args.preview is not None:
        print(f"Preview:     {args.preview}")


if __name__ == "__main__":
    main()
