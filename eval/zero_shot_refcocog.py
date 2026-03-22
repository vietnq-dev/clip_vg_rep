from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from visual_grounding.config import ExperimentConfig
from visual_grounding.data import build_transforms
from visual_grounding.metrics import compute_iou, sanitize_xyxy, xywh2xyxy
from visual_grounding.models import TransVG

from eval.refcocog_parquet import RefCOCOgParquetDataset, collect_parquet_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot RefCOCOg evaluation from parquet shards")
    parser.add_argument(
        "backbone",
        type=str,
        choices=["vit", "convnext", "resnet50", "resnet101", "clip", "clip-vit-b16"],
        help="Model backbone",
    )
    parser.add_argument("-c", "--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["refcocog", "refcocoplus"],
        default="refcocog",
        help="Dataset name (refcocog or refcocoplus)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing parquet shards (defaults by dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split name (val/test/testA/testB)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--use-itm", action="store_true", help="Enable ITM head during eval")
    parser.add_argument(
        "--text-source",
        type=str,
        choices=["answer", "question"],
        default="answer",
        help="Choose text field for the referring expression",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for parquet stability)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.backbone in {"clip", "clip-vit-b16"}:
        cfg.model.text_encoder = "openai/clip-vit-base-patch16"
        cfg.data.max_text_len = 77
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train.device = str(device)
    cfg.train.use_itm = args.use_itm
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    if args.data_dir is None:
        if args.dataset == "refcocoplus":
            args.data_dir = Path("data/raw/RefCOCOplus_hf/data")
        else:
            args.data_dir = Path("data/raw/RefCOCOg_hf/data")

    parquet_paths = collect_parquet_paths(args.data_dir, args.split)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards found for split '{args.split}' in {args.data_dir}")

    _, val_transform = build_transforms(cfg.data.img_size)
    dataset = RefCOCOgParquetDataset(
        parquet_paths=parquet_paths,
        cfg=cfg,
        transform=val_transform,
        text_source=args.text_source,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("{} {} samples: {}", args.dataset, args.split, len(dataset))

    model = TransVG(args.backbone, cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: {}", args.checkpoint)

    total_acc05 = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_bbox = batch["bbox"].to(device)

            output = model(images, input_ids, attention_mask, use_itm=cfg.train.use_itm)
            pred_xywh = output["bbox_pred"] if isinstance(output, dict) else output
            pred_xyxy = sanitize_xyxy(xywh2xyxy(pred_xywh))
            gt_xyxy = sanitize_xyxy(gt_bbox)

            iou_vals = compute_iou(pred_xyxy, gt_xyxy)
            acc05 = (iou_vals >= 0.5).float()
            total_acc05 += acc05.sum().item()
            total_samples += acc05.numel()

    if total_samples == 0:
        raise RuntimeError("No samples found for evaluation")

    mean_acc05 = total_acc05 / total_samples
    std_err_acc05 = math.sqrt(max(mean_acc05 * (1.0 - mean_acc05) / total_samples, 0.0))
    logger.info("Acc@0.5: {:.4f} ± {:.4f}", mean_acc05, std_err_acc05)


if __name__ == "__main__":
    main()
