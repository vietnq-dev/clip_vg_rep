"""Training utilities and experiment runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from loguru import logger
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import create_dataloaders
from .eval import evaluate_model
from .logging_utils import configure_logging
from .metrics import accuracy_at_threshold, compute_iou, giou_loss, sanitize_xyxy, xywh2xyxy, xyxy2xywh
from .models import TransVG

HISTORY_KEYS: Tuple[str, ...] = (
    "train_loss",
    "train_l1",
    "train_giou",
    "train_iou",
    "train_acc@0.25",
    "train_acc@0.5",
    "train_itm_loss",
    "train_itm_acc",
    "val_loss",
    "val_l1",
    "val_giou",
    "val_iou",
    "val_acc@0.25",
    "val_acc@0.5",
    "val_itm_loss",
    "val_itm_acc",
)


def _init_history() -> Dict[str, List[float]]:
    return {key: [] for key in HISTORY_KEYS}


def _load_history_from_file(path: Path) -> Dict[str, List[float]]:
    history = _init_history()
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return history

    for key in HISTORY_KEYS:
        values = payload.get(key)
        if isinstance(values, list):
            history[key] = values
    return history


def _select_amp_dtype(device: str) -> torch.dtype:
    """Pick bf16 if the GPU supports it, else fp16."""
    if device == "cpu":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(backbone: str, cfg: ExperimentConfig) -> TransVG:
    model = TransVG(backbone, cfg)
    return model.to(cfg.train.device)


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.AdamW:
    vision_params = []
    text_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("vision_encoder", "clip_model.vision_model")):
            vision_params.append(param)
        elif name.startswith(("text_encoder", "clip_model.text_model")):
            text_params.append(param)
        else:
            other_params.append(param)

    # Use fused AdamW kernel on CUDA for faster param updates
    use_fused = cfg.train.device == "cuda" and hasattr(torch.optim.AdamW, "fused")
    optimizer = torch.optim.AdamW(
        [
            {"params": vision_params, "lr": cfg.train.lr_backbone},
            {"params": text_params, "lr": cfg.train.lr_backbone},
            {"params": other_params, "lr": cfg.train.lr_new_layers},
        ],
        weight_decay=cfg.train.weight_decay,
        fused=use_fused,
    )
    return optimizer


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_iou: float,
    scaler: GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_iou": best_iou,
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])

    # Move optimizer tensors to the correct device
    for opt_state in optimizer.state.values():
        for key, value in opt_state.items():
            if isinstance(value, torch.Tensor):
                opt_state[key] = value.to(device)

    if scaler is not None and "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])

    epoch = int(state.get("epoch", 0))
    best_iou = float(state.get("best_iou", 0.0))
    return epoch, best_iou


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    model.train()
    l1_loss_fn = nn.L1Loss()
    itm_loss_fn = nn.BCEWithLogitsLoss()
    device = cfg.train.device
    use_amp = cfg.train.use_amp and device == "cuda"
    use_itm = cfg.train.use_itm
    itm_weight = cfg.train.itm_weight

    total_loss = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    total_iou = 0.0
    total_acc25 = 0.0
    total_acc50 = 0.0
    total_itm_loss = 0.0
    total_itm_acc = 0.0

    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        # non_blocking=True overlaps CPU→GPU copy with compute
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        gt_bbox = batch["bbox"].to(device, non_blocking=True)

        # set_to_none=True is faster than filling with zeros
        optimizer.zero_grad(set_to_none=True)

        # Automatic mixed-precision forward pass
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
                loss = loss + itm_weight * itm_l

        # AMP-aware backward + step
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
            optimizer.step()

        # IoU is only for logging — don't track gradients
        with torch.no_grad():
            iou = compute_iou(pred_xyxy.detach(), gt_xyxy).mean()
            acc25 = accuracy_at_threshold(pred_xyxy.detach(), gt_xyxy, 0.25)
            acc50 = accuracy_at_threshold(pred_xyxy.detach(), gt_xyxy, 0.5)

        total_loss += loss.item()
        total_l1 += l1_val.item()
        total_giou += giou_val.item()
        total_iou += iou.item()
        total_acc25 += acc25.item()
        total_acc50 += acc50.item()

        if use_itm:
            total_itm_loss += itm_l.item()
            with torch.no_grad():
                itm_preds = (output["itm_logits"].detach() > 0).float()
                itm_acc = (itm_preds == output["itm_labels"]).float().mean()
            total_itm_acc += itm_acc.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou.item():.4f}", "a@.5": f"{acc50.item():.4f}"})

    num_batches = len(train_loader)
    result = {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "giou": total_giou / num_batches,
        "iou": total_iou / num_batches,
        "acc@0.25": total_acc25 / num_batches,
        "acc@0.5": total_acc50 / num_batches,
        "itm_loss": total_itm_loss / num_batches if use_itm else 0.0,
        "itm_acc": total_itm_acc / num_batches if use_itm else 0.0,
    }
    return result


def train_model(
    backbone: str,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    artifact_dir: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> Tuple[Dict[str, List[float]], TransVG]:

    if cfg.train.device == "cuda":
        torch.backends.cudnn.benchmark = True       # auto-tune conv kernels
        torch.backends.cuda.matmul.allow_tf32 = True # TF32 matmuls (same acc, 3× faster)
        torch.backends.cudnn.allow_tf32 = True       # TF32 cuDNN kernels

    model = build_model(backbone, cfg)
    optimizer = build_optimizer(model, cfg)
    history = _init_history()

    # LR Scheduler
    scheduler = None
    if cfg.train.use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

    # AMP setup
    amp_dtype = _select_amp_dtype(cfg.train.device)
    use_amp = cfg.train.use_amp and cfg.train.device == "cuda"
    
    # GradScaler is only needed for fp16; bf16 doesn't need it
    needs_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler(enabled=True) if needs_scaler else None

    compiled_model: nn.Module = model
    if cfg.train.use_compile:
        logger.info("Compiling model with torch.compile (mode='reduce-overhead')")
        compiled_model = torch.compile(model, mode="reduce-overhead")

    # Artifact paths
    artifact_root = artifact_dir or Path("artifacts")
    # Save each run into a unique folder to avoid overwriting previous runs.
    # Format: ckpt_{backbone}_itm_{weight}_yymmdd_hhmm
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    itm_tag = f"itm_{cfg.train.itm_weight}" if getattr(cfg.train, "use_itm", False) else "itm_0.0"
    run_dir_name = f"ckpt_{backbone}_{itm_tag}_{timestamp}"

    backbone_dir = artifact_root / run_dir_name
    backbone_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = backbone_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = backbone_dir / "history.json"
    best_checkpoint_path = checkpoint_dir / "best.pth"
    last_checkpoint_path = checkpoint_dir / "last.pth"

    logger.info(
        "Saving checkpoints to {checkpoint_dir}",
        checkpoint_dir=checkpoint_dir,
    )

    # Resume logic
    start_epoch = 0
    best_iou = 0.0
    if resume_checkpoint is not None:
        resume_checkpoint_path = Path(resume_checkpoint)
        start_epoch, best_iou = _load_checkpoint(
            resume_checkpoint_path, model, optimizer, cfg.train.device, scaler, scheduler
        )
        logger.info(
            "Resuming backbone {backbone} from {path} starting at epoch {epoch} (target {total})",
            backbone=backbone,
            path=resume_checkpoint_path,
            epoch=start_epoch + 1,
            total=cfg.train.num_epochs,
        )
        if history_path.exists():
            history = _load_history_from_file(history_path)

    # Training loop
    for epoch in range(start_epoch, cfg.train.num_epochs):
        train_stats = train_one_epoch(
            compiled_model, train_loader, optimizer, cfg,
            scaler=scaler, amp_dtype=amp_dtype,
        )
        if scheduler is not None:
            scheduler.step()

        val_stats = evaluate_model(compiled_model, val_loader, cfg)
        epoch_num = epoch + 1

        history["train_loss"].append(train_stats["loss"])
        history["train_l1"].append(train_stats["l1"])
        history["train_giou"].append(train_stats["giou"])
        history["train_iou"].append(train_stats["iou"])
        history["train_acc@0.25"].append(train_stats["acc@0.25"])
        history["train_acc@0.5"].append(train_stats["acc@0.5"])
        history["train_itm_loss"].append(train_stats["itm_loss"])
        history["train_itm_acc"].append(train_stats["itm_acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_l1"].append(val_stats["l1"])
        history["val_giou"].append(val_stats["giou"])
        history["val_iou"].append(val_stats["iou"])
        history["val_acc@0.25"].append(val_stats["acc@0.25"])
        history["val_acc@0.5"].append(val_stats["acc@0.5"])
        history["val_itm_loss"].append(val_stats["itm_loss"])
        history["val_itm_acc"].append(val_stats["itm_acc"])

        itm_log = ""
        if cfg.train.use_itm:
            itm_log = (
                f" itm_loss={train_stats['itm_loss']:.4f} itm_acc={train_stats['itm_acc']:.2f} |"
                f" val_itm_loss={val_stats['itm_loss']:.4f} val_itm_acc={val_stats['itm_acc']:.2f}"
            )
        logger.info(
            "Epoch {epoch}/{total} | train loss={train_loss:.4f} IoU={train_iou:.4f} acc@.25={train_acc25:.2f} acc@.5={train_acc50:.4f} | "
            "val loss={val_loss:.4f} IoU={val_iou:.4f} acc@.25={val_acc25:.2f} acc@.5={val_acc50:.4f}{itm_log}",
            epoch=epoch_num,
            total=cfg.train.num_epochs,
            train_loss=train_stats["loss"],
            train_iou=train_stats["iou"],
            train_acc25=train_stats["acc@0.25"],
            train_acc50=train_stats["acc@0.5"],
            val_loss=val_stats["loss"],
            val_iou=val_stats["iou"],
            val_acc25=val_stats["acc@0.25"],
            val_acc50=val_stats["acc@0.5"],
            itm_log=itm_log,
        )

        # Save state from the unwrapped model (compile wrapper is transparent)
        _save_checkpoint(last_checkpoint_path, model, optimizer, epoch_num, best_iou, scaler, scheduler)

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            _save_checkpoint(best_checkpoint_path, model, optimizer, epoch_num, best_iou, scaler, scheduler)
            logger.success(
                "New best IoU {best:.4f} at epoch {epoch}; stored at {path}",
                best=best_iou,
                epoch=epoch_num,
                path=best_checkpoint_path,
            )

    if best_checkpoint_path.exists():
        state = torch.load(best_checkpoint_path, map_location=cfg.train.device)
        model.load_state_dict(state["model_state"])

    history_path.write_text(json.dumps(history, indent=2))
    logger.info("Persisted history to {path}", path=history_path)

    return history, model


def run_experiment(
    backbones: List[str],
    cfg: ExperimentConfig,
    artifact_dir: Path | None = None,
    run_name: str | None = None,
    resume_checkpoint: Path | None = None,
):
    artifact_root = artifact_dir or Path("artifacts")
    log_dir = artifact_root / "logs"
    log_path = configure_logging(log_dir, run_name)
    logger.info(
        "Starting experiment run; logging to {log_path} and writing artifacts to {artifact_root}",
        log_path=log_path,
        artifact_root=artifact_root,
    )

    if resume_checkpoint is not None and len(backbones) != 1:
        raise ValueError("--resume is only supported when training a single backbone")

    train_loader, val_loader = create_dataloaders(cfg)
    results = {}
    for backbone in backbones:
        logger.info("Training backbone {backbone}", backbone=backbone)
        history, model = train_model(
            backbone,
            cfg,
            train_loader,
            val_loader,
            artifact_root,
            resume_checkpoint=resume_checkpoint,
        )
        results[backbone] = {
            "history": history,
            "model": model,
        }
        logger.info("Finished backbone {backbone}", backbone=backbone)
    return results


__all__ = ["build_model", "train_model", "run_experiment"]
