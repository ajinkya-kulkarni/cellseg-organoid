# cellseg-organoid

StarDist-based organoid instance segmentation using `cellseg-models-pytorch` on a regenerated 256x256 patch version of OrganoIDNetData.

This repo is meant for training and evaluating an instance segmentation model on PDAC organoid microscopy images. The data preparation script pools the original training, validation, and test sources to create a new image-level split balanced by organoid-content properties.

## References

This work is based on the OrganoIDNet/OrganoIDNetData publications and dataset:

- [OrganoIDNet: a deep learning tool for identification of therapeutic effects in PDAC organoid-PBMC co-cultures from time-resolved imaging data](https://link.springer.com/article/10.1007/s13402-024-00958-2)
- [A Curated Cell Life Imaging Dataset of Immune-enriched Pancreatic Cancer Organoids with Pre-trained AI Models](https://www.nature.com/articles/s41597-024-03631-3)

If you use the dataset or reproduce dataset-derived experiments, cite the relevant original publication and follow the upstream dataset terms.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Or use the existing virtual environment if already created:

```bash
source .venv/bin/activate
```

## Remote / Headless Server Setup

On a clean Ubuntu/Debian instance (e.g. a Scaleaway VM) with Python 3.12:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH (or restart your shell)
source ~/.local/bin/env

# Clone the repository
git clone <repo-url> cellseg-organoid
cd cellseg-organoid

# Install dependencies (includes CUDA PyTorch on GPU-equipped instances)
uv sync
```

`uv sync` automatically detects and installs the appropriate PyTorch variant (CUDA on GPU instances, CPU otherwise).

> **Note for headless servers**: OpenCV requires `libGL.so.1` at runtime.
> If you encounter `ImportError: libGL.so.1: cannot open shared object file`,
> install the system library:
>
> ```bash
> sudo apt update && sudo apt install -y libgl1
> ```

## Data Preparation

`prepare_data.py` creates a new 256x256 patch dataset from a source zip containing paired full-field images and instance masks:

- `Images/*.tif` — raw microscopy images
- `Masks/{stem}_mask.tif` — corresponding instance masks

It pools all sources, creates a new 70/15/15 image-level split, and extracts 256x256 patches. Patches from the same source image stay in one split. Stratification uses source-image organoid properties only:

- Organoid instance count
- Foreground mask fraction
- Median instance area

Empty patches are kept. This is intentional so validation and test metrics include background-only behavior.

Run with defaults (expects `OrganoIDNetData-All.zip` on the Desktop and writes `OrganoIDNetData-256`):

```bash
uv run python prepare_data.py
```

Or pass explicit paths:

```bash
uv run python prepare_data.py \
  --source-zip /path/to/OrganoIDNetData-All.zip \
  --output-dir /path/to/OrganoIDNetData-256
```

Warning: `prepare_data.py` removes and recreates the output directory.

Generated layout:

```text
OrganoIDNetData-256/
  Training/
    images/
    labels/
  Validation/
    images/
    labels/
  Test/
    images/
    labels/
  split_manifest.csv
  split_summary.json
```

Labels are MATLAB `.mat` files with:

- `inst`: integer instance labels
- `type`: binary foreground/background labels

## Training

Train on the regenerated dataset:

```bash
uv run python train.py \
  --data-root /path/to/OrganoIDNetData-256 \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 2
```

The model is StarDist with an ImageNet-pretrained `efficientnet_b3` encoder and 32 rays. Training uses:

- BCE-with-logits on the normalized per-instance distance/objectness map
- Masked StarDist MAE on the 32 ray-distance maps, weighted by 0.2
- Weighted foreground/background cross-entropy plus Dice on the auxiliary type head

Validation runs StarDist polygon/NMS post-processing with `score_thresh=0.4` and `iou_thresh=0.4`. Dataset-level PQ is computed at instance IoU > 0.5. `best.pt` and early stopping are based on validation PQ; for a 100-epoch run, patience is 20 epochs.

For a quick smoke test:

```bash
uv run python train.py \
  --data-root /path/to/OrganoIDNetData-256 \
  --epochs 1 \
  --batch-size 2 \
  --sub-size 8 \
  --val-sub-size 32 \
  --num-workers 0
```

Omit `--sub-size` and `--val-sub-size` for the real run so the full training and validation splits are used.

Note: `train.py` removes and recreates the `output/` and `models/` directories before starting, so old checkpoints and visualizations are cleared.

Outputs:

- `models/last.pt`
- `models/best.pt` (best validation PQ)
- `output/viz_epoch_*.png` (input, ground truth instances, predicted instances)

## Evaluation

Evaluate the best checkpoint on all test patches:

```bash
uv run python evaluate.py --data-root /path/to/OrganoIDNetData-256
```

Or choose a checkpoint explicitly:

```bash
uv run python evaluate.py --data-root /path/to/OrganoIDNetData-256 --ckpt models/best.pt
```

Evaluation uses the same StarDist post-processing defaults as validation (`score_thresh=0.4`, NMS IoU `0.4`) and reports binary Dice/IoU, count error, instance precision/recall/F1, and dataset-level PQ/DQ/SQ at instance IoU > 0.5.

Note: `evaluate.py` removes and recreates the `output/` directory before writing results.

Outputs:

- `output/metrics.csv`
- `output/pred_*.png` (input, ground truth instances, predicted instances)

For empty ground-truth patches, perfect empty predictions score Dice and IoU as `1.0`.

## Notes

This is a regenerated split, not the original benchmark split from OrganoIDNetData. It intentionally mixes the original training, validation, and test sources before creating a new stratified dataset.

The repository license applies to this code. Dataset usage is governed by the upstream OrganoIDNetData release and associated publications.

## License

Apache License 2.0. See `LICENSE`.
