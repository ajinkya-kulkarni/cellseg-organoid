"""Lightweight StarDist polygon reconstruction and NMS for deployment.

This module intentionally has no PyTorch or cellseg-models-pytorch dependency.

Polygon rendering utilities are adapted from the official StarDist project:
Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
BSD 3-Clause License.

NMS structure is adapted from cellseg-models-pytorch's StarDist postprocessing.
"""

import math

import numpy as np
from numba import njit
from scipy.spatial import KDTree
from skimage.draw import polygon
from skimage.morphology import disk, erosion

from config import NMS_IOU_THRESH, SCORE_THRESH


@njit(cache=True, fastmath=True)
def _intersection(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height


@njit
def _get_bboxes(dist: np.ndarray, points: np.ndarray):
    n_polys = dist.shape[0]
    n_rays = dist.shape[1]

    bbox_x1 = np.zeros(n_polys)
    bbox_x2 = np.zeros(n_polys)
    bbox_y1 = np.zeros(n_polys)
    bbox_y2 = np.zeros(n_polys)
    areas = np.zeros(n_polys)

    angle_pi = 2 * math.pi / n_rays
    max_dist = 0.0

    for i in range(n_polys):
        max_radius_outer = 0.0
        py = points[i, 0]
        px = points[i, 1]

        for k in range(n_rays):
            d = dist[i, k]
            y = py + d * np.sin(angle_pi * k)
            x = px + d * np.cos(angle_pi * k)

            if k == 0:
                bbox_x1[i] = x
                bbox_x2[i] = x
                bbox_y1[i] = y
                bbox_y2[i] = y
            else:
                bbox_x1[i] = min(x, bbox_x1[i])
                bbox_x2[i] = max(x, bbox_x2[i])
                bbox_y1[i] = min(y, bbox_y1[i])
                bbox_y2[i] = max(y, bbox_y2[i])

            max_radius_outer = max(d, max_radius_outer)

        areas[i] = (bbox_x2[i] - bbox_x1[i]) * (bbox_y2[i] - bbox_y1[i])
        max_dist = max(max_dist, max_radius_outer)

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2, areas, max_dist


@njit
def _suppress_bbox(
    query: np.ndarray,
    current_idx: int,
    boxes: np.ndarray,
    areas: np.ndarray,
    suppressed: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    for i in range(len(query)):
        query_idx = query[i]

        if suppressed[query_idx]:
            continue

        overlap = _intersection(boxes[current_idx], boxes[query_idx])
        iou = overlap / min(
            areas[current_idx] + 1e-10,
            areas[query_idx] + 1e-10,
        )
        suppressed[query_idx] = iou > iou_threshold

    return suppressed


def _nms_stardist(
    boxes: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    max_dist: float,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    keep = []
    kdtree = KDTree(points, leafsize=16)
    suppressed = np.full(len(boxes), False)

    for current_idx in range(len(scores)):
        if suppressed[current_idx]:
            continue
        if scores[current_idx] < score_threshold:
            break

        query = np.asarray(
            kdtree.query_ball_point(points[current_idx], max_dist),
            dtype=np.int64,
        )
        suppressed = _suppress_bbox(
            query,
            current_idx,
            boxes,
            areas,
            suppressed,
            iou_threshold,
        )
        keep.append(current_idx)

    return np.asarray(keep, dtype=np.int64)


def _ray_angles(n_rays: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def _dist_to_coord(dist: np.ndarray, points: np.ndarray) -> np.ndarray:
    n_rays = dist.shape[1]
    phis = _ray_angles(n_rays)
    coord = (
        dist[:, np.newaxis]
        * np.array([np.sin(phis), np.cos(phis)])
    ).astype(np.float32)
    coord += points[..., np.newaxis]
    return coord


def _polygons_to_label_coord(
    coord: np.ndarray,
    shape: tuple[int, int],
    labels: np.ndarray,
) -> np.ndarray:
    result = np.zeros(shape, np.int32)

    for label, coordinates in zip(labels, coord):
        rr, cc = polygon(*coordinates, shape)
        result[rr, cc] = label + 1

    return result


def _polygons_to_label(
    dist: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    points = points[order]
    dist = dist[order]

    coord = _dist_to_coord(dist, points)
    return _polygons_to_label_coord(coord, shape=shape, labels=order)


def post_proc_stardist(
    probability: np.ndarray,
    ray_map: np.ndarray,
    score_thresh: float = SCORE_THRESH,
    iou_thresh: float = NMS_IOU_THRESH,
    trim_bboxes: bool = True,
) -> np.ndarray:
    """Convert dense StarDist outputs into a 2D int32 instance-label mask."""
    probability = np.asarray(probability, dtype=np.float32)
    ray_map = np.asarray(ray_map, dtype=np.float32)

    if probability.ndim != 2:
        raise ValueError(
            f"probability must have shape (H, W), got {probability.shape}"
        )
    if ray_map.ndim != 3:
        raise ValueError(
            f"ray_map must have shape (R, H, W), got {ray_map.shape}"
        )
    if ray_map.shape[1:] != probability.shape:
        raise ValueError(
            "ray_map spatial shape must match probability: "
            f"{ray_map.shape[1:]} vs {probability.shape}"
        )

    shape = probability.shape
    dist = ray_map.transpose(1, 2, 0)

    mask = (probability > score_thresh).astype(np.int32)

    # Match the cellseg-models-pytorch StarDist deployment path.
    border = np.zeros_like(mask)
    border[2:-2, 2:-2] = 1
    mask *= border

    if trim_bboxes:
        mask -= erosion(mask, disk(2))

    points = np.stack(np.where(mask), axis=1)
    dist = dist[mask > 0]
    scores = probability[mask > 0]

    order = np.argsort(scores)[::-1]
    dist = dist[order]
    scores = scores[order]
    points = points[order]

    x1, y1, x2, y2, areas, max_dist = _get_bboxes(dist, points)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms_stardist(
        boxes,
        points,
        scores,
        areas,
        max_dist,
        score_threshold=score_thresh,
        iou_threshold=iou_thresh,
    )

    return _polygons_to_label(
        dist[keep],
        points[keep],
        scores[keep],
        shape=shape,
    )
