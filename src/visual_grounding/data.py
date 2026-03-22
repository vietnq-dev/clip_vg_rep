"""Dataset utilities for RefCOCO stored in LMDB."""
from __future__ import annotations

import io
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from transformers import AutoTokenizer

from .config import ExperimentConfig


Transform = Optional[Callable[[Image.Image], torch.Tensor]]


def cache_lmdb_keys(lmdb_path: Path, output_path: Optional[Path] = None) -> List[bytes]:
    """Cache LMDB keys so we only iterate the database once."""
    if output_path is None:
        output_path = lmdb_path.with_suffix(lmdb_path.suffix + ".keys.pkl")

    if output_path.exists():
        with open(output_path, "rb") as f:
            keys = pickle.load(f)
        return keys

    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, subdir=False
    )
    keys: List[bytes] = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = pickle.loads(value)
            if isinstance(data, dict):
                keys.append(key)
    env.close()

    with open(output_path, "wb") as f:
        pickle.dump(keys, f)
    return keys


class RefCOCODatasetLMDB(Dataset):
    """PyTorch dataset that streams RefCOCO samples from an LMDB file."""

    def __init__(self, lmdb_path: Path, cfg: ExperimentConfig, transform: Transform = None):
        self.lmdb_path = lmdb_path
        self.cfg = cfg
        self.transform = transform
        self.env: Optional[lmdb.Environment] = None

        self.keys = cache_lmdb_keys(lmdb_path)
        self.length = len(self.keys)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)
        self.max_length = cfg.data.max_text_len
        if hasattr(self.tokenizer, "model_max_length") and self.tokenizer.model_max_length:
            if self.tokenizer.model_max_length < self.max_length:
                self.max_length = self.tokenizer.model_max_length

    def _init_env(self) -> None:
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, subdir=False
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._init_env()
        key = self.keys[idx]
        assert self.env is not None
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Key {key!r} not found in {self.lmdb_path}")
            data = pickle.loads(value)

        img = Image.open(io.BytesIO(data["img"])).convert("RGB")
        orig_w, orig_h = img.size

        sents = data["sents"]
        text = random.choice(sents) if isinstance(sents, list) else sents

        mask = Image.open(io.BytesIO(data["mask"]))
        mask_array = np.array(mask)
        ys, xs = np.where(mask_array > 0)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h

        # Bbox in absolute coordinates [x1, y1, x2, y2]
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        if self.transform:
            img_out = self.transform(img)
            bbox_out = torch.tensor(
                [x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h], dtype=torch.float32
            )
        else:
            img_out = img
            bbox_out = torch.tensor(
                [x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h], dtype=torch.float32
            )

        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": img_out,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "bbox": bbox_out,
            "text": text,
            "img_name": data.get("img_name", "")
        }


def build_transforms(img_size: int) -> Tuple[Transform, Transform]:
    """Return train/val transforms that keep bbox alignments."""
    transform = T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform, transform


def create_datasets(
    cfg: ExperimentConfig,
    train_transform: Transform = None,
    val_transform: Transform = None,
) -> Tuple[RefCOCODatasetLMDB, RefCOCODatasetLMDB]:
    """Instantiate train/val datasets."""
    if train_transform is None or val_transform is None:
        default_train, default_val = build_transforms(cfg.data.img_size)
        train_transform = train_transform or default_train
        val_transform = val_transform or default_val

    train_dataset = RefCOCODatasetLMDB(
        cfg.data.lmdb_dir / "train.lmdb", cfg, train_transform
    )
    val_dataset = RefCOCODatasetLMDB(
        cfg.data.lmdb_dir / "val.lmdb", cfg, val_transform
    )
    return train_dataset, val_dataset


def create_dataloaders(
    cfg: ExperimentConfig,
    train_transform: Transform = None,
    val_transform: Transform = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders that can be reused across experiments."""
    train_dataset, val_dataset = create_datasets(cfg, train_transform, val_transform)

    use_workers = cfg.data.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,                     # avoid tiny last batch → steady GPU util
        persistent_workers=use_workers,      # keep workers alive between epochs
        prefetch_factor=cfg.data.prefetch_factor if use_workers else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
        prefetch_factor=cfg.data.prefetch_factor if use_workers else None,
    )

    return train_loader, val_loader


__all__ = [
    "cache_lmdb_keys",
    "RefCOCODatasetLMDB",
    "build_transforms",
    "create_datasets",
    "create_dataloaders",
]
