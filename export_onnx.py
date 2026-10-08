"""Export a trained OrganoIDNet StarDist checkpoint to ONNX.

The ONNX graph contains only the dense neural-network forward pass:
  image -> binary_map logits + ray_map

Sigmoid activation, StarDist polygon reconstruction, and NMS stay outside the graph.
Spatial dimensions are fixed at 256x256; the batch dimension is dynamic.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from config import ENCODER, N_RAYS, PATCH_SIZE
from utils import create_stardist_model

OPSET_VERSION = 18


class StarDistONNXWrapper(nn.Module):
    """Flatten the structured StarDist output into ONNX-friendly tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nuc = self.model(x)["nuc"]
        if nuc.binary_map is None:
            raise RuntimeError("StarDist export requires a binary_map output head.")
        return nuc.binary_map, nuc.aux_map


def _check_export_requirements() -> None:
    """Validate the PyTorch version and ONNX exporter dependencies."""
    try:
        major, minor = (
            int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Unable to determine PyTorch version from {torch.__version__!r}."
        ) from None

    if (major, minor) < (2, 5):
        raise RuntimeError(
            f"ONNX export requires PyTorch >= 2.5; found {torch.__version__}."
        )

    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        missing.append("onnxscript")

    if missing:
        packages = " ".join(missing)
        raise ImportError(
            "Missing ONNX export dependencies. Run with: "
            f"`uv run --with {packages.replace(' ', ' --with ')} python export_onnx.py ...`"
        )


def load_checkpoint(checkpoint_path: Path) -> nn.Module:
    """Load the trained EfficientNet-B3 StarDist network on CPU."""
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


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    dynamic_batch: bool = True,
) -> Path:
    """Export the trained StarDist dense prediction network to ONNX."""
    _check_export_requirements()

    model = load_checkpoint(checkpoint_path)
    wrapper = StarDistONNXWrapper(model).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(
        (1, 3, PATCH_SIZE, PATCH_SIZE),
        dtype=torch.float32,
        device="cpu",
    )

    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(output_path),
            input_names=["image"],
            output_names=["binary_map", "ray_map"],
            opset_version=OPSET_VERSION,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )

    # Structural validation of the generated model.
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export OrganoIDNet StarDist best.pt to ONNX"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(__file__).parent / "models" / "best.pt",
        help="PyTorch checkpoint (default: models/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models" / "best.onnx",
        help="Output ONNX path (default: models/best.onnx)",
    )
    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export batch size 1 instead of a dynamic batch dimension",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    output = export_onnx(
        checkpoint_path=args.ckpt,
        output_path=args.output,
        dynamic_batch=not args.static_batch,
    )

    print(f"Exported: {output}")
    print("Input:       image      [batch, 3, 256, 256] float32")
    print("Output:      binary_map [batch, 1, 256, 256] raw logits")
    print("Output:      ray_map    [batch, 32, 256, 256] raw distances")
    print(f"ONNX opset:  {OPSET_VERSION}")
    print(f"Batch axis:  {'static (1)' if args.static_batch else 'dynamic'}")
    print("ONNX checker: PASS")


if __name__ == "__main__":
    main()
