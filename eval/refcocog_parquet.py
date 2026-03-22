from __future__ import annotations

import io
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from visual_grounding.config import ExperimentConfig


@dataclass(frozen=True)
class RowGroupIndex:
    file_idx: int
    row_group: int
    start: int
    length: int


class RefCOCOgParquetDataset(Dataset):
    """Dataset reader for RefCOCOg parquet shards (LMMS format)."""

    def __init__(
        self,
        parquet_paths: Sequence[Path],
        cfg: ExperimentConfig,
        transform=None,
        text_source: str = "answer",
        image_root: Optional[Path] = None,
    ) -> None:
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.cfg = cfg
        self.transform = transform
        self.text_source = text_source
        self.image_root = image_root

        self._parquet_files = [pq.ParquetFile(str(p)) for p in self.parquet_paths]
        self._row_groups: List[RowGroupIndex] = []
        self._length = 0
        for file_idx, pf in enumerate(self._parquet_files):
            for rg_idx in range(pf.num_row_groups):
                nrows = pf.metadata.row_group(rg_idx).num_rows
                self._row_groups.append(RowGroupIndex(file_idx, rg_idx, self._length, nrows))
                self._length += nrows

        self._cache_key: Optional[Tuple[int, int]] = None
        self._cache_table = None

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)
        self.max_length = cfg.data.max_text_len
        if hasattr(self.tokenizer, "model_max_length") and self.tokenizer.model_max_length:
            if self.tokenizer.model_max_length < self.max_length:
                self.max_length = self.tokenizer.model_max_length

    def __len__(self) -> int:
        return self._length

    def _find_row_group(self, idx: int) -> Tuple[RowGroupIndex, int]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        starts = [rg.start for rg in self._row_groups]
        pos = bisect_right(starts, idx) - 1
        rg = self._row_groups[pos]
        offset = idx - rg.start
        return rg, offset

    def _read_row(self, rg: RowGroupIndex, offset: int) -> Dict[str, Any]:
        cache_key = (rg.file_idx, rg.row_group)
        if self._cache_key != cache_key:
            pf = self._parquet_files[rg.file_idx]
            self._cache_table = pf.read_row_group(
                rg.row_group,
                columns=["image", "question", "answer", "bbox", "file_name"],
            )
            self._cache_key = cache_key
        table = self._cache_table
        row = table.slice(offset, 1).to_pylist()[0]
        return row

    def _load_image(self, image_struct: Dict[str, Any]) -> Image.Image:
        image_bytes = image_struct.get("bytes")
        if image_bytes:
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_path = image_struct.get("path")
        if image_path is None:
            raise ValueError("Image bytes missing and no image path provided.")

        path = Path(image_path)
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        return Image.open(path).convert("RGB")

    def _select_text(self, row: Dict[str, Any]) -> str:
        if self.text_source == "answer":
            answers = row.get("answer") or []
            if isinstance(answers, list) and len(answers) > 0:
                return answers[0]
        return row.get("question", "")

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rg, offset = self._find_row_group(idx)
        row = self._read_row(rg, offset)

        img = self._load_image(row["image"])
        orig_w, orig_h = img.size

        x, y, w, h = row["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox = torch.tensor(
            [x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h], dtype=torch.float32
        )

        if self.transform:
            img_out = self.transform(img)
        else:
            img_out = img

        text = self._select_text(row)
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
            "bbox": bbox,
            "text": text,
            "img_name": row.get("file_name", ""),
        }


def collect_parquet_paths(data_dir: Path, split: str) -> List[Path]:
    direct = sorted(data_dir.glob(f"{split}-*.parquet"))
    if direct:
        return direct

    if split in {"testA", "testB", "test"}:
        test_shards = sorted(data_dir.glob("test-*.parquet"))
        if split == "test":
            return test_shards
        if len(test_shards) >= 2:
            return [test_shards[0]] if split == "testA" else [test_shards[1]]

    return []
