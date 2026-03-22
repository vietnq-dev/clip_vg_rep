"""Command-line entry point for running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual grounding models with uv artifacts")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["vit", "convnext"],
        help="Backbone identifiers to train (e.g., vit convnext)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for logs, checkpoints, and histories",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the run; defaults to timestamp",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override the number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the training batch size",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint (single backbone only)",
    )
    parser.add_argument(
        "--use-itm",
        action="store_true",
        default=False,
        help="Enable image-text matching auxiliary loss",
    )
    parser.add_argument(
        "--itm-weight",
        type=float,
        default=0.5,
        help="Weight for the ITM loss term (default: 0.5)",
    )
    parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        default=False,
        help="Enable cosine annealing learning rate scheduler",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Hugging Face id (or local dir) of the CLIP backbone",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint to load into the CLIP model (e.g., FineCLIP)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()

    cfg.model.clip_model_name = args.clip_model_name
    cfg.model.clip_checkpoint_path = args.clip_checkpoint

    clip_backbones = {"clip", "clip-vit-b16"}
    uses_clip = any(backbone.lower() in clip_backbones for backbone in args.backbones)
    if uses_clip and len(args.backbones) > 1:
        raise ValueError("CLIP backbones require a single-backbone run due to tokenizer compatibility")
    if uses_clip:
        cfg.model.text_encoder = args.clip_model_name
        cfg.data.max_text_len = 77

    if args.num_epochs is not None:
        cfg.train.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    cfg.train.use_itm = args.use_itm
    cfg.train.itm_weight = args.itm_weight
    cfg.train.use_cosine_scheduler = args.use_cosine_scheduler

    run_experiment(
        args.backbones,
        cfg,
        artifact_dir=args.artifacts_dir,
        run_name=args.run_name,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
