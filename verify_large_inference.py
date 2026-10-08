"""Verify that tiled large-image inference preserves 256x256 patch results."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from large_inference import predict_large_image
from onnx_inference import OrganoidONNXPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare patch inference with tiled inference on Validation patches"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--num-patches", type=int, default=32)
    args = parser.parse_args()

    if args.num_patches <= 0:
        raise ValueError("--num-patches must be greater than zero")

    image_dir = args.data_root / "Validation" / "images"
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Validation PNGs found in: {image_dir}")

    n = min(args.num_patches, len(paths))
    indices = np.linspace(0, len(paths) - 1, n, dtype=int)
    paths = [paths[i] for i in indices]

    predictor = OrganoidONNXPredictor(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )

    exact = 0
    differing_pixels = 0
    total_pixels = 0

    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        direct = predictor.predict_patch(image_rgb)
        tiled = predict_large_image(
            predictor,
            image_rgb,
            overlap=64,
            batch_size=1,
        )

        same = np.array_equal(direct, tiled)
        if same:
            exact += 1
        else:
            differing_pixels += int(np.count_nonzero(direct != tiled))
        total_pixels += direct.size

        print(f"  {'PASS' if same else 'DIFF'}  {path.name}")

    agreement = 1.0 - differing_pixels / max(total_pixels, 1)

    print()
    print(f"Patches checked:          {len(paths)}")
    print(f"Exact masks:              {exact}/{len(paths)}")
    print(f"Instance pixel agreement: {agreement:.8f}")

    if exact != len(paths):
        raise SystemExit(1)

    print()
    print("Patch <-> tiled parity: PASS")


if __name__ == "__main__":
    main()
