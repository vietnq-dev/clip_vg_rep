"""Evaluation helpers and visualization tools."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .metrics import accuracy_at_threshold, compute_iou, giou_loss, sanitize_xyxy, xywh2xyxy, xyxy2xywh


@torch.inference_mode()  # faster than no_grad — disables view-tracking & versioning
def evaluate_model(model: nn.Module, data_loader: DataLoader, cfg: ExperimentConfig) -> Dict[str, float]:
    model.eval()
    l1_loss_fn = nn.L1Loss()
    itm_loss_fn = nn.BCEWithLogitsLoss()
    device = cfg.train.device
    use_amp = getattr(cfg.train, "use_amp", False) and device == "cuda"
    use_itm = getattr(cfg.train, "use_itm", False)
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    total_loss = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    total_iou = 0.0
    total_acc25 = 0
    total_acc50 = 0
    total_itm_loss = 0.0
    total_itm_acc = 0.0
    total_samples = 0

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        gt_bbox = batch["bbox"].to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            output = model(images, input_ids, attention_mask, use_itm=use_itm)

            if use_itm:
                pred_xywh = output["bbox_pred"]
            else:
                pred_xywh = output

            pred_xyxy = sanitize_xyxy(xywh2xyxy(pred_xywh))
            gt_xyxy = sanitize_xyxy(gt_bbox)

            pred_xywh_clean = xyxy2xywh(pred_xyxy)
            gt_xywh = xyxy2xywh(gt_xyxy)

            l1_val = l1_loss_fn(pred_xywh_clean, gt_xywh)
            giou_val = giou_loss(pred_xyxy, gt_xyxy)
            loss = l1_val + 2.0 * giou_val

            if use_itm:
                itm_l = itm_loss_fn(output["itm_logits"], output["itm_labels"])
                total_itm_loss += itm_l.item()
                itm_preds = (output["itm_logits"] > 0).float()
                total_itm_acc += (itm_preds == output["itm_labels"]).float().mean().item()

        iou_per_sample = compute_iou(pred_xyxy, gt_xyxy)
        bs = iou_per_sample.size(0)
        total_loss += loss.item()
        total_l1 += l1_val.item()
        total_giou += giou_val.item()
        total_iou += iou_per_sample.sum().item()
        total_acc25 += (iou_per_sample >= 0.25).sum().item()
        total_acc50 += (iou_per_sample >= 0.5).sum().item()
        total_samples += bs

    num_batches = len(data_loader)
    num_samples = total_samples or len(data_loader.dataset)
    return {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "giou": total_giou / num_batches,
        "iou": total_iou / num_samples,
        "acc@0.25": total_acc25 / num_samples,
        "acc@0.5": total_acc50 / num_samples,
        "itm_loss": total_itm_loss / num_batches if use_itm else 0.0,
        "itm_acc": total_itm_acc / num_batches if use_itm else 0.0,
    }


@torch.no_grad()
def visualize_predictions(
    models: Mapping[str, nn.Module],
    dataset: Dataset,
    cfg: ExperimentConfig,
    num_samples: int = 4,
    save_path: Optional[Path] = None,
) -> None:
    for model in models.values():
        model.to(cfg.train.device)
        model.eval()

    fig, axes = plt.subplots(num_samples, len(models) + 1, figsize=(5 * (len(models) + 1), 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for row in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(cfg.train.device)
        input_ids = sample["input_ids"].unsqueeze(0).to(cfg.train.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(cfg.train.device)
        gt_bbox = sample["bbox"].numpy()

        img = sample["image"].permute(1, 2, 0).numpy()
        img = np.clip(img * std + mean, 0, 1)

        ax = axes[row, 0]
        ax.imshow(img)
        x1, y1, x2, y2 = gt_bbox * cfg.data.img_size
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", facecolor="none"))
        ax.set_title(f"GT\n{sample['text'][:40]}", fontsize=10)
        ax.axis("off")

        for col, (name, model) in enumerate(models.items(), start=1):
            pred_xywh = model(image_tensor, input_ids, attention_mask)[0]
            pred_xyxy = sanitize_xyxy(xywh2xyxy(pred_xywh.unsqueeze(0)))[0].cpu().numpy()
            pred_iou = compute_iou(
                torch.tensor([pred_xyxy]), torch.tensor([gt_bbox])
            )[0].item()

            ax_model = axes[row, col]
            ax_model.imshow(img)
            px1, py1, px2, py2 = pred_xyxy * cfg.data.img_size
            ax_model.add_patch(
                patches.Rectangle((px1, py1), px2 - px1, py2 - py1, linewidth=2, edgecolor="orange", facecolor="none")
            )
            ax_model.set_title(f"{name}\nIoU: {pred_iou:.3f}", fontsize=10)
            ax_model.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


__all__ = ["evaluate_model", "visualize_predictions"]
