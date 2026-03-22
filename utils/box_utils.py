"""Bounding box utility functions."""
import torch


def xyxy2xywh(box):
    """Convert box from [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
    x1, y1, x2, y2 = box.unbind(-1) if box.dim() > 1 else box
    return torch.stack([
        (x1 + x2) / 2,  # x_center
        (y1 + y2) / 2,  # y_center
        x2 - x1,        # width
        y2 - y1         # height
    ], dim=-1)


def xywh2xyxy(box):
    """Convert box from [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box.unbind(-1) if box.dim() > 1 else box
    return torch.stack([
        cx - w / 2,  # x1
        cy - h / 2,  # y1
        cx + w / 2,  # x2
        cy + h / 2   # y2
    ], dim=-1)
