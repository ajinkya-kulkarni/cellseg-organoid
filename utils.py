"""Shared utilities for OrganoIDNet training and evaluation."""

import torch
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_nuclei

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def create_stardist_model(
    enc_name="efficientnet_b3",
    enc_pretrain=True,
    n_rays=32,
):
    return stardist_nuclei(
        n_rays=n_rays,
        n_nuc_classes=2,
        enc_name=enc_name,
        enc_pretrain=enc_pretrain,
    )
