"""Shared utilities for OrganoIDNet training and evaluation."""

import torch
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def create_cellpose_model(enc_name="efficientnet_b0", enc_pretrain=True):
    return cellpose_nuclei(n_nuc_classes=2, enc_name=enc_name, enc_pretrain=enc_pretrain)
