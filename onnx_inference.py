"""Standalone ONNX inference for one 256x256 organoid image patch.

This module intentionally imports neither torch nor cellseg-models-pytorch.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from stardist_postproc import post_proc_stardist

PATCH_SIZE = 256
N_RAYS = 32
SCORE_THRESH = 0.4
NMS_IOU_THRESH = 0.4


def normalize_patch(image: np.ndarray) -> np.ndarray:
    """Match training-time global min-max normalization to [0, 1]."""
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    expected = (PATCH_SIZE, PATCH_SIZE, 3)
    if image.shape != expected:
        raise ValueError(f"Expected image shape {expected}, got {image.shape}")

    normalized = cv2.normalize(
        image.astype(np.float32),
        None,
        alpha=0.0,
        beta=1.0,
        norm_type=cv2.NORM_MINMAX,
    )
    return np.ascontiguousarray(normalized)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0

    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)

    return out


class OrganoidONNXPredictor:
    """ONNX Runtime predictor for 256x256 StarDist organoid patches."""

    def __init__(self, model_path: str | Path, providers=None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for inference. "
                "Install it with `uv add onnxruntime` or run with "
                "`uv run --with onnxruntime ...`."
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        kwargs = {}
        if providers is not None:
            kwargs["providers"] = providers

        self.session = ort.InferenceSession(str(self.model_path), **kwargs)

        input_info = self.session.get_inputs()
        output_names = [output.name for output in self.session.get_outputs()]

        if len(input_info) != 1 or input_info[0].name != "image":
            raise RuntimeError("Unexpected ONNX input contract; expected input 'image'.")
        if output_names != ["binary_map", "ray_map"]:
            raise RuntimeError(
                "Unexpected ONNX output contract; expected "
                "['binary_map', 'ray_map'], got "
                f"{output_names}"
            )

    def predict_batch_logits(
        self,
        images_rgb: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return raw binary logits (B,H,W) and rays (B,32,H,W)."""
        if not images_rgb:
            raise ValueError("images_rgb must contain at least one patch")

        batch = np.stack(
            [
                normalize_patch(image).transpose(2, 0, 1)
                for image in images_rgb
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        binary_logits, ray_map = self.session.run(
            ["binary_map", "ray_map"],
            {"image": batch},
        )

        logits = np.asarray(binary_logits[:, 0], dtype=np.float32)
        rays = np.asarray(ray_map, dtype=np.float32)

        expected_logits = (len(images_rgb), PATCH_SIZE, PATCH_SIZE)
        expected_rays = (len(images_rgb), N_RAYS, PATCH_SIZE, PATCH_SIZE)
        if logits.shape != expected_logits:
            raise RuntimeError(f"Unexpected binary-logit shape: {logits.shape}")
        if rays.shape != expected_rays:
            raise RuntimeError(f"Unexpected ray-map shape: {rays.shape}")

        return logits, rays

    def predict_dense(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return probability (H,W) and ray distances (32,H,W)."""
        logits, rays = self.predict_batch_logits([image_rgb])
        probability = sigmoid(logits[0])
        return probability, rays[0]

    def predict_patch(
        self,
        image_rgb: np.ndarray,
        score_thresh: float = SCORE_THRESH,
        nms_iou_thresh: float = NMS_IOU_THRESH,
    ) -> np.ndarray:
        """Return a 256x256 int32 instance-label mask."""
        probability, rays = self.predict_dense(image_rgb)
        labels = post_proc_stardist(
            probability,
            rays,
            score_thresh=score_thresh,
            iou_thresh=nms_iou_thresh,
            trim_bboxes=True,
        )
        return labels.astype(np.int32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone ONNX StarDist inference on one 256x256 patch"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/onnx_patch_labels.npy"),
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = OrganoidONNXPredictor(args.model)
    labels = predictor.predict_patch(image_rgb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)

    print(f"Saved: {args.output}")
    print(f"Shape: {labels.shape}")
    print(f"dtype: {labels.dtype}")
    print(f"Instances: {len(np.unique(labels)) - 1}")


if __name__ == "__main__":
    main()
