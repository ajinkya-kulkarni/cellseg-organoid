"""Shared model and inference contract.

Keep this module dependency-free so both training code and lightweight ONNX
deployment code can import the same constants.
"""

ENCODER = "efficientnet_b3"
N_RAYS = 32
PATCH_SIZE = 256

SCORE_THRESH = 0.4
NMS_IOU_THRESH = 0.4
INSTANCE_IOU_THRESH = 0.5

TILE_OVERLAP = 64
