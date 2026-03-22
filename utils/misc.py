"""Miscellaneous utility functions."""
import torch
import torch.nn.functional as F


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    Wrapper around torch.nn.functional.interpolate.
    Equivalent to nn.functional.interpolate but supports empty batch sizes.
    """
    if input.numel() == 0:
        return input
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
