from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from visual_grounding.config import ExperimentConfig
from visual_grounding.data import RefCOCODatasetLMDB, build_transforms
from visual_grounding.metrics import compute_iou, sanitize_xyxy, xywh2xyxy
from visual_grounding.models import TransVG


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return img


def _predict_boxes(model: TransVG, sample: dict, cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = torch.device(cfg.train.device)
    image = sample['image'].unsqueeze(0).to(device)
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image, input_ids, attention_mask, use_itm=cfg.train.use_itm)
        pred_xywh = output['bbox_pred'] if isinstance(output, dict) else output

    pred_xyxy = sanitize_xyxy(xywh2xyxy(pred_xywh)).cpu()
    gt_xyxy = sanitize_xyxy(sample['bbox'].unsqueeze(0)).cpu()
    iou = compute_iou(pred_xyxy, gt_xyxy)[0].item()
    return pred_xyxy[0], gt_xyxy[0], iou


def _draw_prediction(ax, image_np: np.ndarray, pred_box: torch.Tensor, gt_box: torch.Tensor, text: str, iou: float) -> None:
    ax.imshow(image_np)
    height, width = image_np.shape[:2]

    def to_pixels(box: torch.Tensor) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box.tolist()
        return x1 * width, y1 * height, x2 * width, y2 * height

    px1, py1, px2, py2 = to_pixels(pred_box)
    gx1, gy1, gx2, gy2 = to_pixels(gt_box)
    ax.add_patch(patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
    ax.add_patch(patches.Rectangle((px1, py1), px2 - px1, py2 - py1, linewidth=2, edgecolor='orange', facecolor='none', label='Pred'))
    ax.set_title(f"{text}\nIoU: {iou:.3f}", fontsize=11)
    ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right')


def export_visualizations(
    model: TransVG,
    dataset: RefCOCODatasetLMDB,
    cfg: ExperimentConfig,
    backbone: str,
    split: str,
    count: int,
    output_dir: Path,
    dpi: int,
    seed: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    if total == 0:
        logger.warning('Dataset is empty, skipping visualization')
        return

    num_samples = min(count, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=num_samples, replace=False) if num_samples < total else np.arange(total)

    for order, dataset_idx in enumerate(sorted(indices.tolist())):
        sample = dataset[dataset_idx]
        pred_box, gt_box, iou = _predict_boxes(model, sample, cfg)
        image_np = _denormalize_image(sample['image'])

        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_prediction(ax, image_np, pred_box, gt_box, sample['text'], iou)

        save_path = output_dir / f"{backbone}_{split}_{dataset_idx:06d}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info('Saved visualization %d/%d → %s', order + 1, num_samples, save_path)


def main():
    parser = argparse.ArgumentParser(description='Visual Grounding Inference')
    parser.add_argument(
        'backbone',
        type=str,
        choices=['vit', 'convnext', 'resnet50', 'resnet101', 'clip', 'clip-vit-b16'],
        help='Vision backbone: vit, convnext, resnet50, resnet101, or clip (clip-vit-b16)'
    )
    parser.add_argument('-c', '--checkpoint', type=Path, required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--split', type=str, default='val',
                        help='LMDB split to evaluate (default: val)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override evaluation batch size')
    parser.add_argument('--use-itm', action='store_true',
                        help='Enable ITM head (match training config)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Skip visualization step (visualization ON by default)')
    parser.set_defaults(visualize=True)
    parser.add_argument('--vis-samples', type=int, default=10,
                        help='Number of individual images to export (default: 10)')
    parser.add_argument('--vis-dir', type=Path, default=Path('outputs/predictions'),
                        help='Directory to store visualization files')
    parser.add_argument('--vis-dpi', type=int, default=300,
                        help='Image DPI for publication-ready figures (default: 300)')
    parser.add_argument('--vis-seed', type=int, default=None,
                        help='Optional RNG seed for selecting visualization samples')
    args = parser.parse_args()
    
    config = ExperimentConfig()
    if args.backbone in {'clip', 'clip-vit-b16'}:
        config.model.text_encoder = 'openai/clip-vit-base-patch16'
        config.data.max_text_len = 77
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.train.device = str(device)
    config.train.use_itm = args.use_itm
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    logger.info(f'Device: {device}')

    # Data
    lmdb_path = config.data.lmdb_dir / f'{args.split}.lmdb'
    if not lmdb_path.exists():
        raise FileNotFoundError(f'LMDB split not found: {lmdb_path}')

    _, val_transform = build_transforms(config.data.img_size)
    val_dataset = RefCOCODatasetLMDB(lmdb_path, config, val_transform)
    use_workers = config.data.num_workers > 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=use_workers,
        prefetch_factor=config.data.prefetch_factor if use_workers else None,
    )
    logger.info(f'{args.split} samples: {len(val_loader.dataset)}')

    # Model
    model = TransVG(args.backbone, config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if 'model_state' in state:
        state = state['model_state']
    elif 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    logger.info(f'Loaded: {args.checkpoint}')

    # Evaluate
    total_acc05 = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gt_bbox = batch['bbox'].to(device)

            output = model(images, input_ids, attention_mask, use_itm=config.train.use_itm)
            if isinstance(output, dict):
                pred_xywh = output['bbox_pred']
            else:
                pred_xywh = output
            pred_xyxy = sanitize_xyxy(xywh2xyxy(pred_xywh))
            gt_xyxy = sanitize_xyxy(gt_bbox)

            iou_vals = compute_iou(pred_xyxy, gt_xyxy)
            acc05 = (iou_vals >= 0.5).float()
            total_acc05 += acc05.sum().item()
            total_samples += acc05.numel()

    if total_samples == 0:
        raise RuntimeError('No samples found for evaluation')

    mean_acc05 = total_acc05 / total_samples
    # Standard error of the mean accuracy (Bernoulli proportion)
    std_err_acc05 = math.sqrt(max(mean_acc05 * (1.0 - mean_acc05) / total_samples, 0.0))
    logger.info(f'Acc@0.5: {mean_acc05:.4f} ± {std_err_acc05:.4f}')

    # Visualize
    if args.visualize:
        export_visualizations(
            model,
            val_loader.dataset,
            config,
            backbone=args.backbone,
            split=args.split,
            count=args.vis_samples,
            output_dir=args.vis_dir,
            dpi=args.vis_dpi,
            seed=args.vis_seed,
        )


if __name__ == '__main__':
    main()