"""Visualization utilities (single-panel overlay GT+Pred, headless-safe)."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # safe on servers (no DISPLAY needed)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visual_grounding.metrics import compute_iou, xywh2xyxy, sanitize_xyxy


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalized CHW tensor -> HWC float in [0,1]."""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0.0, 1.0)


def _to_xyxy_norm_1x4(box: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> torch.Tensor:
    """Convert a single bbox to torch.Tensor of shape (1,4), float32. Assumes xyxy normalized."""
    if isinstance(box, torch.Tensor):
        return box.detach().float().view(1, 4).cpu()
    arr = np.asarray(box, dtype=np.float32).reshape(1, 4)
    return torch.from_numpy(arr)


def _extract_pred_bbox_xywh(out: Any) -> torch.Tensor:
    """
    Robustly extract predicted bbox (cx,cy,w,h) from model output.

    Supports:
      - dict with keys like 'pred_bbox', 'pred_boxes', ...
      - tuple/list (uses first element)
      - tensor directly (assumes it's the bbox)
    Returns a tensor of shape (B,4) or (4,)
    """
    if isinstance(out, dict):
        for k in ("pred_bbox", "pred_boxes", "pred_box", "boxes", "bbox"):
            if k in out:
                return out[k]
        raise KeyError(f"Model output is dict but no bbox key found. Keys={list(out.keys())}")

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise ValueError("Model output is an empty tuple/list.")
        return out[0]

    if torch.is_tensor(out):
        return out

    raise TypeError(f"Unsupported model output type: {type(out)}")


def visualize_predictions(
    model: torch.nn.Module,
    dataset: Any,
    config: Any,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    show: bool = False,
    backbone_name: Optional[str] = None,
) -> Optional[str]:
    """
    Visualize predictions by overlaying GT (green) and Pred (blue) on the SAME image.

    Expects sample keys:
      - 'image': torch.Tensor (C,H,W), ImageNet-normalized
      - 'input_ids', 'attention_mask'
      - 'bbox': xyxy normalized in [0,1]
      - 'text': str
    Model output:
      - bbox in cxcywh normalized (either dict['pred_bbox'] or tensor/tuple)
    """
    import matplotlib.font_manager as fm
    font_prop = fm.FontProperties(fname="assets/fonts/lmroman7-regular.otf", size=16)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    n = int(num_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # flatten for easy indexing
    axes_flat = axes.reshape(-1)

    for i in range(n):
        ax = axes_flat[i]

        idx = random.randint(0, len(dataset) - 1)
        sample: Dict[str, Any] = dataset[idx]

        image = sample["image"].unsqueeze(0).to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        img = denormalize_image(sample["image"])
        H, W = img.shape[0], img.shape[1]

        # GT bbox (xyxy normalized)
        gt_xyxy_t = sanitize_xyxy(_to_xyxy_norm_1x4(sample["bbox"]))
        gt_xyxy = gt_xyxy_t[0].numpy()

        # Pred bbox
        with torch.no_grad():
            out = model(image, input_ids, attention_mask)
            pred_xywh_all = _extract_pred_bbox_xywh(out)
            if pred_xywh_all.dim() == 2:
                pred_xywh = pred_xywh_all[0]
            elif pred_xywh_all.dim() == 1:
                pred_xywh = pred_xywh_all
            elif pred_xywh_all.dim() == 3 and pred_xywh_all.size(-1) == 4 and pred_xywh_all.size(1) == 1:
                pred_xywh = pred_xywh_all[0, 0]
            else:
                raise ValueError(f"Unexpected pred bbox shape: {tuple(pred_xywh_all.shape)}")

            pred_xyxy_t = sanitize_xyxy(xywh2xyxy(pred_xywh.view(1, 4).float().cpu()))
            pred_xyxy = pred_xyxy_t[0].numpy()

        # IoU
        iou = compute_iou(
            torch.tensor(pred_xyxy, dtype=torch.float32).unsqueeze(0),
            torch.tensor(gt_xyxy, dtype=torch.float32).unsqueeze(0),
        )[0].item()

        # Plot image
        ax.imshow(img)

        # Draw GT (green)
        x1, y1, x2, y2 = gt_xyxy
        ax.add_patch(
            patches.Rectangle(
                (x1 * W, y1 * H),
                max(1.0, (x2 - x1) * W),
                max(1.0, (y2 - y1) * H),
                linewidth=2.5,
                edgecolor="green",
                facecolor="none",
            )
        )

        # Draw Pred (blue)
        px1, py1, px2, py2 = pred_xyxy
        ax.add_patch(
            patches.Rectangle(
                (px1 * W, py1 * H),
                max(1.0, (px2 - px1) * W),
                max(1.0, (py2 - py1) * H),
                linewidth=2.5,
                edgecolor="dodgerblue",
                facecolor="none",
            )
        )

        text = sample.get("text", "")
        short_text = text[:55] + ("..." if len(text) > 55 else "")
        bb = backbone_name or "backbone"
        ax.set_title(f"{bb} | IoU={iou:.3f} | {short_text}", fontproperties=font_prop)
        ax.axis("off")

    # turn off unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()

    saved_abs: Optional[str] = None
    if save_path:
        outp = Path(save_path)
        if outp.suffix.lower() != ".pdf":
            outp = outp.with_suffix(".pdf")
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, bbox_inches="tight", format="pdf")
        saved_abs = str(outp.resolve())
        print(f"[visualize_predictions] Saved: {saved_abs}")

    if show and os.environ.get("DISPLAY"):
        plt.show()

    plt.close(fig)
    return saved_abs


def plot_training_curves(histories, names):
    """Plot training curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for history, name in zip(histories, names):
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{name} Train")
        axes[0].plot(epochs, history["val_loss"], "--", label=f"{name} Val")
        axes[1].plot(epochs, history["train_iou"], label=f"{name} Train")
        axes[1].plot(epochs, history["val_iou"], "--", label=f"{name} Val")

    for ax, title, ylabel in zip(axes, ["Loss", "IoU"], ["Loss", "IoU"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    plt.close(fig)
