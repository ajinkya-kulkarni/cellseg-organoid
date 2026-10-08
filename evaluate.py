"""Evaluate best.pt on all test patches, print metrics, save CSV, generate 5 viz plots."""

import argparse
import csv
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from cellseg_models_pytorch.metrics.binary_metrics import iou_score
from cellseg_models_pytorch.metrics.functional import pairwise_pixel_stats
from cellseg_models_pytorch.postproc import post_proc_stardist
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization
from cellseg_models_pytorch.utils import remap_label
from scipy.io import loadmat
from tqdm import tqdm

from config import ENCODER, INSTANCE_IOU_THRESH, NMS_IOU_THRESH, SCORE_THRESH
from utils import DEVICE, create_stardist_model

OUT = Path(__file__).parent / "output" / "eval"
TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])
IOU_THRESH = INSTANCE_IOU_THRESH
N_VIZ = 5


def load_model(ckpt_path):
    model = create_stardist_model(enc_name=ENCODER, enc_pretrain=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, img_rgb):
    img_norm = TRANSFORM(image=img_rgb)["image"]
    tensor = (
        torch.from_numpy(img_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    out = model(tensor)

    prob = torch.sigmoid(out["nuc"].binary_map[0, 0])
    rays = out["nuc"].aux_map[0]

    instances = post_proc_stardist(
        prob.cpu().numpy(),
        rays.cpu().numpy(),
        score_thresh=SCORE_THRESH,
        iou_thresh=NMS_IOU_THRESH,
        trim_bboxes=True,
        normalize=False,
    )

    return instances


def pq_stats(gt, pred, iou_thresh=IOU_THRESH):
    """Return TP, FP, FN and matched-IoU sum for dataset-level PQ."""
    gt = remap_label(gt)
    pred = remap_label(pred)

    n_gt = len(np.unique(gt)) - 1
    n_pred = len(np.unique(pred)) - 1

    iou = pairwise_pixel_stats(gt, pred, metric_func=iou_score)
    if not iou:
        return 0, n_pred, n_gt, 0.0

    iou = iou[0]
    matches = iou > iou_thresh

    tp = int(matches.sum())
    fp = n_pred - tp
    fn = n_gt - tp
    iou_sum = float(iou[matches].sum())

    return tp, fp, fn, iou_sum


def dice_iou(pred_mask, gt_mask):
    inter = (pred_mask & gt_mask).sum()
    denominator = pred_mask.sum() + gt_mask.sum()
    union = (pred_mask | gt_mask).sum()
    if denominator == 0:
        return 1.0, 1.0
    dice = 2 * inter / (denominator + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def visualize(img_rgb, gt_inst, pred_inst, save_path, title=""):
    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    if gt_inst is not None:
        gt_n = len(np.unique(gt_inst)) - 1
        gt_viz = gt_inst.astype(float)
        if gt_viz.max() > 0:
            gt_viz = 0.08 + 0.75 * gt_viz / gt_viz.max()
        gt_viz[gt_inst == 0] = np.nan
        axes[1].imshow(gt_viz, cmap=spec, vmin=0, vmax=1)
        axes[1].set_title(f"GT ({gt_n} organoids)")
    else:
        axes[1].set_title("No GT")
    axes[1].axis("off")

    pred_n = len(np.unique(pred_inst)) - 1
    pred_viz = pred_inst.astype(float)
    if pred_viz.max() > 0:
        pred_viz = 0.08 + 0.75 * pred_viz / pred_viz.max()
    pred_viz[pred_inst == 0] = np.nan
    axes[2].imshow(pred_viz, cmap=spec, vmin=0, vmax=1)
    axes[2].set_title(f"Pred ({pred_n} organoids)")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def print_section(
    name,
    n,
    dice_arr,
    iou_arr,
    ce_arr,
    gt_arr,
    pred_arr,
    tp,
    fp,
    fn,
    pq_tp,
    pq_fp,
    pq_fn,
    pq_iou_sum,
):
    if n == 0:
        return
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    dq = pq_tp / (pq_tp + 0.5 * pq_fp + 0.5 * pq_fn + 1e-8)
    sq = pq_iou_sum / (pq_tp + 1e-8)
    pq = dq * sq
    print(f"\n{'='*55}")
    print(f"  {name} ({n} patches)")
    print(f"{'='*55}")
    print("  Segmentation:")
    print(f"    Dice:  {dice_arr.mean():.4f} ± {dice_arr.std():.4f}  (median {np.median(dice_arr):.4f})")
    print(f"    IoU:   {iou_arr.mean():.4f} ± {iou_arr.std():.4f}  (median {np.median(iou_arr):.4f})")
    print(f"    Dice < 0.1: {(dice_arr < 0.1).sum()} / {n}")
    print("  Detection:")
    print(f"    GT count:   {gt_arr.mean():.1f} ± {gt_arr.std():.1f}")
    print(f"    Pred count: {pred_arr.mean():.1f} ± {pred_arr.std():.1f}")
    print(f"    Bias:       {pred_arr.mean() - gt_arr.mean():.1f}")
    print(f"    |Error|:    {ce_arr.mean():.2f} ± {ce_arr.std():.2f}  (median {np.median(ce_arr):.1f})")
    print(f"  Instance matching @ IoU > {IOU_THRESH}:")
    print(f"    Precision:  {p:.4f}")
    print(f"    Recall:     {r:.4f}")
    print(f"    F1:         {f:.4f}")
    print("  Panoptic quality:")
    print(f"    PQ:         {pq:.4f}")
    print(f"    DQ:         {dq:.4f}")
    print(f"    SQ:         {sq:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best.pt on test patches")
    parser.add_argument("--data-root", type=str, required=True, help="Path to OrganoIDNetData-256")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (default: ../models/best.pt)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ckpt = Path(args.ckpt) if args.ckpt else Path(__file__).parent / "models" / "best.pt"

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Run train.py first.")
        return

    model = load_model(ckpt)
    print(f"Loaded: {ckpt}")

    img_dir = data_root / "Test" / "images"
    label_dir = data_root / "Test" / "labels"
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No test images found in: {img_dir}")

    rows = []
    dice_list, iou_list = [], []
    ce_list, gt_n_list, pred_n_list = [], [], []
    tp_total, fp_total, fn_total = 0, 0, 0
    tp_sp = {"Human": 0, "Mouse": 0}
    fp_sp = {"Human": 0, "Mouse": 0}
    fn_sp = {"Human": 0, "Mouse": 0}

    pq_tp_total, pq_fp_total, pq_fn_total = 0, 0, 0
    pq_iou_total = 0.0
    pq_tp_sp = {"Human": 0, "Mouse": 0}
    pq_fp_sp = {"Human": 0, "Mouse": 0}
    pq_fn_sp = {"Human": 0, "Mouse": 0}
    pq_iou_sp = {"Human": 0.0, "Mouse": 0.0}

    random.seed(42)
    viz_paths = set(random.sample(img_paths, min(N_VIZ, len(img_paths))))

    for img_path in tqdm(img_paths, desc="Evaluating"):
        stem = img_path.stem
        species = "Human" if "Human" in stem else "Mouse"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = label_dir / img_path.with_suffix(".mat").name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        mat = loadmat(str(label_path))
        gt_inst = mat["inst"]

        pred_inst = predict(model, img_rgb)

        if img_path in viz_paths:
            save_path = OUT / f"pred_{stem}.png"
            visualize(img_rgb, gt_inst, pred_inst, save_path, title=stem)

        gt_bin = gt_inst > 0
        pred_bin = pred_inst > 0
        d, i = dice_iou(pred_bin, gt_bin)
        dice_list.append(d)
        iou_list.append(i)

        gt_n = len(np.unique(gt_inst)) - 1
        pred_n = len(np.unique(pred_inst)) - 1
        ce_list.append(abs(pred_n - gt_n))
        gt_n_list.append(gt_n)
        pred_n_list.append(pred_n)

        pq_tp, pq_fp, pq_fn, pq_iou_sum = pq_stats(
            gt_inst,
            pred_inst,
            iou_thresh=IOU_THRESH,
        )

        # At strict IoU > 0.5 for disjoint instance masks, valid matches are
        # inherently one-to-one, so DQ uses the same TP/FP/FN as detection F1.
        tp, fp, fn = pq_tp, pq_fp, pq_fn
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tp_sp[species] += tp
        fp_sp[species] += fp
        fn_sp[species] += fn
        pq_tp_total += pq_tp
        pq_fp_total += pq_fp
        pq_fn_total += pq_fn
        pq_iou_total += pq_iou_sum
        pq_tp_sp[species] += pq_tp
        pq_fp_sp[species] += pq_fp
        pq_fn_sp[species] += pq_fn
        pq_iou_sp[species] += pq_iou_sum

        rows.append({
            "patch": stem, "species": species,
            "dice": round(d, 4), "iou": round(i, 4),
            "gt_count": gt_n, "pred_count": pred_n, "count_error": abs(pred_n - gt_n),
            "tp": tp, "fp": fp, "fn": fn,
        })

    csv_path = OUT / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    dice_arr = np.array(dice_list)
    iou_arr = np.array(iou_list)
    ce_arr = np.array(ce_list)
    gt_arr = np.array(gt_n_list)
    pred_arr = np.array(pred_n_list)

    print_section(
        "Overall",
        len(rows),
        dice_arr,
        iou_arr,
        ce_arr,
        gt_arr,
        pred_arr,
        tp_total,
        fp_total,
        fn_total,
        pq_tp_total,
        pq_fp_total,
        pq_fn_total,
        pq_iou_total,
    )
    for sp in ["Human", "Mouse"]:
        mask = np.array([r["species"] == sp for r in rows])
        print_section(
            sp,
            mask.sum(),
            dice_arr[mask],
            iou_arr[mask],
            ce_arr[mask],
            gt_arr[mask],
            pred_arr[mask],
            tp_sp[sp],
            fp_sp[sp],
            fn_sp[sp],
            pq_tp_sp[sp],
            pq_fp_sp[sp],
            pq_fn_sp[sp],
            pq_iou_sp[sp],
        )

    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
