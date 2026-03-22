"""Bounding-box helper functions and metrics."""
from __future__ import annotations

import torch
import torch.nn as nn


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(pred_bbox[:, 0], gt_bbox[:, 0])
    y1 = torch.max(pred_bbox[:, 1], gt_bbox[:, 1])
    x2 = torch.min(pred_bbox[:, 2], gt_bbox[:, 2])
    y2 = torch.min(pred_bbox[:, 3], gt_bbox[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = box_area_xyxy(pred_bbox)
    gt_area = box_area_xyxy(gt_bbox)
    union = pred_area + gt_area - intersection + 1e-6
    return intersection / union


def accuracy_at_threshold(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor, threshold: float) -> torch.Tensor:
    """Fraction of samples whose IoU ≥ *threshold*."""
    iou = compute_iou(pred_bbox, gt_bbox)
    return (iou >= threshold).float().mean()


def giou_loss(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(pred_bbox[:, 0], gt_bbox[:, 0])
    y1 = torch.max(pred_bbox[:, 1], gt_bbox[:, 1])
    x2 = torch.min(pred_bbox[:, 2], gt_bbox[:, 2])
    y2 = torch.min(pred_bbox[:, 3], gt_bbox[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area_pred = box_area_xyxy(pred_bbox)
    area_gt = box_area_xyxy(gt_bbox)
    union = area_pred + area_gt - inter + 1e-6
    iou = inter / union

    cx1 = torch.min(pred_bbox[:, 0], gt_bbox[:, 0])
    cy1 = torch.min(pred_bbox[:, 1], gt_bbox[:, 1])
    cx2 = torch.max(pred_bbox[:, 2], gt_bbox[:, 2])
    cy2 = torch.max(pred_bbox[:, 3], gt_bbox[:, 3])
    c_area = torch.clamp(cx2 - cx1, min=0) * torch.clamp(cy2 - cy1, min=0) + 1e-6
    giou = iou - (c_area - union) / c_area
    return (1 - giou).mean()

__all__ = [
    "box_area_xyxy",
    "sanitize_xyxy",
    "xyxy2xywh",
    "xywh2xyxy",
    "compute_iou",
    "accuracy_at_threshold",
    "giou_loss",
]
