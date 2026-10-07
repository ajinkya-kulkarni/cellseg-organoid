#!/usr/bin/env python3
"""Train StarDist (cellseg_models.pytorch) on OrganoIDNetData-256."""

import os
import math
import random
import shutil
from pathlib import Path

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

import albumentations as A
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

from utils import DEVICE, create_stardist_model
from cellseg_models_pytorch.losses import MAE
from cellseg_models_pytorch.metrics.binary_metrics import iou_score
from cellseg_models_pytorch.metrics.functional import pairwise_pixel_stats
from cellseg_models_pytorch.postproc import post_proc_stardist
from cellseg_models_pytorch.transforms.functional import gen_dist_maps, gen_stardist_maps
from cellseg_models_pytorch.utils import remap_label

SEED = 42

OUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = Path(__file__).parent / "models"

LR = 1e-3
ENCODER = "efficientnet_b3"
N_RAYS = 32
ETA_MIN = 1e-6
CLIP_NORM = 1.0


class OrganoidDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_paths = sorted(Path(img_dir).glob("*.png"))
        self.label_paths = sorted(Path(label_dir).glob("*.mat"))

        if len(self.img_paths) != len(self.label_paths):
            raise RuntimeError(
                f"Count mismatch: {len(self.img_paths)} images vs {len(self.label_paths)} labels"
            )

        for img_p, label_p in zip(self.img_paths, self.label_paths):
            if img_p.stem != label_p.stem:
                raise RuntimeError(
                    f"Mismatch: image '{img_p.name}' paired with label '{label_p.name}'"
                )

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat = loadmat(str(self.label_paths[idx]))
        inst = mat["inst"].astype(np.int32)
        type_map = mat["type"].astype(np.int32)

        if self.transform:
            aug = self.transform(image=img, masks=[inst, type_map])
            img = aug["image"]
            inst = aug["masks"][0].astype(np.int32)
            type_map = aug["masks"][1].astype(np.int32)

        rays = gen_stardist_maps(inst, N_RAYS).astype(np.float32)
        dist = gen_dist_maps(inst).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        inst_map = torch.from_numpy(inst).long()
        type_map = torch.from_numpy(type_map).long()
        rays = torch.from_numpy(rays).float()
        dist = torch.from_numpy(dist).float()

        return {
            "image": img,
            "inst": inst_map,
            "type": type_map,
            "rays": rays,
            "dist": dist,
        }


def dice_score(logits, target, eps=1e-8):
    pred = logits.argmax(dim=1)
    pred_fg = pred == 1
    target_fg = target == 1
    intersection = (pred_fg & target_fg).sum().float()
    denominator = pred_fg.sum().float() + target_fg.sum().float()
    if denominator.item() == 0:
        return 1.0
    return (2.0 * intersection / (denominator + eps)).item()


def dice_loss(logits, target, eps=1e-8):
    """Differentiable Dice loss for the foreground class."""
    prob_fg = torch.softmax(logits, dim=1)[:, 1]
    target_fg = (target == 1).float()
    intersection = (prob_fg * target_fg).sum()
    denominator = prob_fg.sum() + target_fg.sum()
    if denominator.item() == 0:
        return torch.tensor(0.0, device=logits.device)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def pq_stats(gt, pred, iou_thresh=0.5):
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


@torch.no_grad()
def save_viz(loader, model, epoch, path):
    model.eval()
    batch = next(iter(loader))
    img = batch["image"][0].cpu()
    gt_inst = batch["inst"][0].cpu().numpy()

    out = model(img.unsqueeze(0).to(DEVICE))
    prob_fg = torch.softmax(out["nuc"].type_map, dim=1)[0, 1].cpu().numpy()

    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img.permute(1, 2, 0).clip(0, 1))
    axes[0].set_title("Input")
    axes[0].axis("off")

    gt_viz = gt_inst.astype(float)
    if gt_viz.max() > 0:
        gt_viz = 0.08 + 0.75 * gt_viz / gt_viz.max()
    gt_viz[gt_inst == 0] = np.nan
    axes[1].imshow(gt_viz, cmap=spec, vmin=0, vmax=1)
    axes[1].set_title("GT instances")
    axes[1].axis("off")

    axes[2].imshow(prob_fg, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title(f"Pred probability (epoch {epoch})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train StarDist on OrganoIDNetData-256")
    parser.add_argument("--data-root", type=str, required=True, help="Path to OrganoIDNetData-256")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--sub-size", type=int, default=None,
                        help="Limit training samples (default: use all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.epochs
    sub_size = args.sub_size
    NUM_WORKERS = args.num_workers

    WARMUP_EPOCHS = max(3, MAX_EPOCHS // 10)
    PATIENCE = max(5, MAX_EPOCHS // 5)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=10, fill_mask=0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
            MinMaxNormalization(always_apply=True),
        ]
    )
    val_transform = A.Compose(
        [
            MinMaxNormalization(always_apply=True),
        ]
    )

    train_ds = OrganoidDataset(
        DATA_ROOT / "Training" / "images",
        DATA_ROOT / "Training" / "labels",
        transform=train_transform,
    )
    val_ds = OrganoidDataset(
        DATA_ROOT / "Validation" / "images",
        DATA_ROOT / "Validation" / "labels",
        transform=val_transform,
    )

    if sub_size is not None:
        train_ds = Subset(train_ds, range(min(sub_size, len(train_ds))))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = create_stardist_model(
        enc_name=ENCODER,
        enc_pretrain=True,
        n_rays=N_RAYS,
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    ray_loss_fn = MAE(alpha=1e-4, apply_mask=True)

    # Compute class weights from training labels
    n_bg, n_fg = 0, 0
    for p in tqdm(
        sorted((DATA_ROOT / "Training" / "labels").glob("*.mat")),
        desc="Computing class weights",
        leave=False,
    ):
        m = loadmat(str(p))
        t = m["type"].ravel()
        n_bg += (t == 0).sum()
        n_fg += (t == 1).sum()
    total = n_bg + n_fg
    class_weights = torch.tensor(
        [total / (2 * n_bg), total / (2 * n_fg)], dtype=torch.float32, device=DEVICE
    )
    print(f"  Class weights: bg={class_weights[0]:.4f}, fg={class_weights[1]:.4f}")

    best_pq = -1.0
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # LR schedule: linear warmup → cosine decay
        if epoch <= WARMUP_EPOCHS:
            lr = LR * epoch / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)
            lr = ETA_MIN + (LR - ETA_MIN) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img = batch["image"].to(DEVICE)
            gt_rays = batch["rays"].to(DEVICE)
            gt_dist = batch["dist"].to(DEVICE)
            gt_type = batch["type"].to(DEVICE)

            out = model(img)

            fg_mask = (gt_type > 0).float()
            dist_loss = F.binary_cross_entropy_with_logits(
                out["nuc"].binary_map[:, 0],
                gt_dist,
            )
            ray_loss = ray_loss_fn(
                out["nuc"].aux_map,
                gt_rays,
                mask=fg_mask,
            )
            type_loss = (
                F.cross_entropy(out["nuc"].type_map, gt_type, weight=class_weights)
                + dice_loss(out["nuc"].type_map, gt_type)
            )
            loss = dist_loss + 0.2 * ray_loss + type_loss

            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        pq_tp = 0
        pq_fp = 0
        pq_fn = 0
        pq_iou_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                gt_rays = batch["rays"].to(DEVICE)
                gt_dist = batch["dist"].to(DEVICE)
                gt_type = batch["type"].to(DEVICE)

                out = model(img)

                fg_mask = (gt_type > 0).float()
                dist_loss = F.binary_cross_entropy_with_logits(
                    out["nuc"].binary_map[:, 0],
                    gt_dist,
                )
                ray_loss = ray_loss_fn(
                    out["nuc"].aux_map,
                    gt_rays,
                    mask=fg_mask,
                )
                type_loss = (
                    F.cross_entropy(
                        out["nuc"].type_map, gt_type, weight=class_weights
                    )
                    + dice_loss(out["nuc"].type_map, gt_type)
                )
                val_loss += (
                    dist_loss.item()
                    + 0.2 * ray_loss.item()
                    + type_loss.item()
                )
                val_dice += dice_score(out["nuc"].type_map, gt_type)

                gt_inst = batch["inst"].cpu().numpy()
                pred_dist = out["nuc"].binary_map[:, 0].cpu().numpy()
                pred_rays = out["nuc"].aux_map.cpu().numpy()

                for i in range(img.shape[0]):
                    pred_inst = post_proc_stardist(
                        pred_dist[i],
                        pred_rays[i],
                        score_thresh=0.4,
                        iou_thresh=0.4,
                        trim_bboxes=True,
                        normalize=True,
                    )
                    tp, fp, fn, iou_sum = pq_stats(
                        gt_inst[i],
                        pred_inst,
                        iou_thresh=0.5,
                    )
                    pq_tp += tp
                    pq_fp += fp
                    pq_fn += fn
                    pq_iou_sum += iou_sum

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        dq = pq_tp / (pq_tp + 0.5 * pq_fp + 0.5 * pq_fn + 1e-8)
        sq = pq_iou_sum / (pq_tp + 1e-8)
        avg_pq = dq * sq

        print(
            f"  train: {avg_train:.4f}  val: {avg_val:.4f}  "
            f"dice: {avg_dice:.4f}  pq: {avg_pq:.4f}  "
            f"dq: {dq:.4f}  sq: {sq:.4f}  lr: {lr:.2e}"
        )

        torch.save(
            {"model": model.state_dict(), "epoch": epoch}, MODELS_DIR / "last.pt"
        )

        if avg_pq > best_pq:
            best_pq = avg_pq
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "pq": avg_pq,
                    "dq": dq,
                    "sq": sq,
                    "dice": avg_dice,
                },
                MODELS_DIR / "best.pt",
            )
            print(
                f"  -> best.pt "
                f"(pq={avg_pq:.4f}, dq={dq:.4f}, sq={sq:.4f})"
            )
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            save_viz(val_loader, model, epoch, OUT_DIR / f"viz_epoch_{epoch}.png")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE})")
            break

    print(f"\nDone. Best PQ: {best_pq:.4f}")


if __name__ == "__main__":
    main()
