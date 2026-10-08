"""Verify PyTorch and ONNX parity on real validation patches.

Checks:
1. Dense binary logits are numerically close.
2. Dense StarDist ray maps are numerically close.
3. Final postprocessed instance masks are pixel-identical.

Both PyTorch and ONNX Runtime run on CPU to isolate export/runtime differences.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from cellseg_models_pytorch.postproc import post_proc_stardist as reference_postproc
from scipy.special import expit

from onnx_inference import OrganoidONNXPredictor, normalize_patch
from stardist_postproc import post_proc_stardist as deployment_postproc
from utils import create_stardist_model

ENCODER = "efficientnet_b3"
N_RAYS = 32
PATCH_SIZE = 256
SCORE_THRESH = 0.4
NMS_IOU_THRESH = 0.4
RTOL = 1e-4
ATOL = 2e-4

def load_pytorch_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a 'model' state dict: {checkpoint_path}")

    model = create_stardist_model(
        enc_name=ENCODER,
        enc_pretrain=False,
        n_rays=N_RAYS,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def load_patch_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def reference_instances(binary_logits: np.ndarray, ray_map: np.ndarray) -> np.ndarray:
    probability = expit(binary_logits).astype(np.float32, copy=False)
    return reference_postproc(
        probability,
        ray_map,
        score_thresh=SCORE_THRESH,
        iou_thresh=NMS_IOU_THRESH,
        trim_bboxes=True,
        normalize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify best.pt and best.onnx on Validation patches"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(__file__).parent / "models" / "best.pt",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).parent / "models" / "best.onnx",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=32,
        help="Number of evenly spaced Validation patches to compare (default: 32)",
    )
    args = parser.parse_args()

    if args.num_patches <= 0:
        raise ValueError("--num-patches must be greater than zero")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Run with: "
            "`uv run --with onnxruntime python verify_onnx.py ...`"
        ) from exc

    image_dir = args.data_root / "Validation" / "images"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Validation PNGs found in: {image_dir}")

    n_check = min(args.num_patches, len(image_paths))
    sample_indices = np.linspace(0, len(image_paths) - 1, n_check, dtype=int)
    image_paths = [image_paths[i] for i in sample_indices]
    images_rgb = [load_patch_rgb(path) for path in image_paths]
    batch = np.stack(
        [
            normalize_patch(image).transpose(2, 0, 1)
            for image in images_rgb
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    model = load_pytorch_model(args.ckpt)
    with torch.inference_mode():
        output = model(torch.from_numpy(batch))["nuc"]
        pt_binary = output.binary_map.cpu().numpy()
        pt_rays = output.aux_map.cpu().numpy()

    predictor = OrganoidONNXPredictor(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )
    ort_binary, ort_rays = predictor.session.run(
        ["binary_map", "ray_map"],
        {"image": batch},
    )

    binary_abs = np.abs(pt_binary - ort_binary)
    rays_abs = np.abs(pt_rays - ort_rays)

    binary_max = float(binary_abs.max())
    binary_mean = float(binary_abs.mean())
    rays_max = float(rays_abs.max())
    rays_mean = float(rays_abs.mean())

    dense_ok = True
    try:
        np.testing.assert_allclose(
            ort_binary,
            pt_binary,
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            ort_rays,
            pt_rays,
            rtol=RTOL,
            atol=ATOL,
        )
    except AssertionError:
        dense_ok = False

    exact_instance_matches = 0
    differing_pixels = 0
    total_pixels = 0

    for index, path in enumerate(image_paths):
        pt_inst = reference_instances(pt_binary[index, 0], pt_rays[index])

        probability = expit(ort_binary[index, 0]).astype(np.float32, copy=False)
        ort_inst = deployment_postproc(
            probability,
            ort_rays[index],
            score_thresh=SCORE_THRESH,
            iou_thresh=NMS_IOU_THRESH,
            trim_bboxes=True,
        )

        same = np.array_equal(pt_inst, ort_inst)
        if same:
            exact_instance_matches += 1
        else:
            differing_pixels += int(np.count_nonzero(pt_inst != ort_inst))
        total_pixels += pt_inst.size

        status = "PASS" if same else "DIFF"
        print(f"  {status}  {path.name}")

    pixel_agreement = 1.0 - differing_pixels / max(total_pixels, 1)
    instance_ok = exact_instance_matches == len(image_paths)

    print()
    print(f"Patches checked:          {len(image_paths)}")
    print(f"Binary logits max abs:    {binary_max:.3e}")
    print(f"Binary logits mean abs:   {binary_mean:.3e}")
    print(f"Ray map max abs:          {rays_max:.3e}")
    print(f"Ray map mean abs:         {rays_mean:.3e}")
    print(f"Dense allclose:           {'PASS' if dense_ok else 'FAIL'}")
    print(
        "Exact instance masks:    "
        f"{exact_instance_matches}/{len(image_paths)} "
        f"({'PASS' if instance_ok else 'FAIL'})"
    )
    print(f"Instance pixel agreement: {pixel_agreement:.8f}")

    if not dense_ok or not instance_ok:
        raise SystemExit(1)

    print()
    print("PT <-> ONNX parity: PASS")


if __name__ == "__main__":
    main()
