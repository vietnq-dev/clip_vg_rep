"""Configuration objects for the visual grounding research project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """All dataset and dataloader related hyper-parameters."""
    data_dir: Path = Path("./data")
    lmdb_dir: Path = Path("./data/lmdb/refcoco")
    img_size: int = 224
    max_text_len: int = 40
    num_workers: int = 4
    prefetch_factor: int = 4


@dataclass
class TrainingConfig:
    """Optimizer and schedule settings."""
    batch_size: int = 64
    num_epochs: int = 20
    lr_backbone: float = 1e-5
    lr_new_layers: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True                # automatic mixed-precision (bf16/fp16, no quant)
    use_compile: bool = True            # torch.compile graph-level fusion
    use_itm: bool = False               # image-text matching auxiliary loss
    itm_weight: float = 1.0             # weight for the ITM loss term
    use_cosine_scheduler: bool = False  # use cosine annealing learning rate scheduler


@dataclass
class ModelConfig:
    """Model architecture hyper-parameters."""
    hidden_dim: int = 256
    text_encoder: str = "bert-base-uncased"
    vl_num_layers: int = 4
    vl_num_heads: int = 8
    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_checkpoint_path: Path | None = None


@dataclass
class ExperimentConfig:
    """Bundle data/model/training configs for convenience."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
