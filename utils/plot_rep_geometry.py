#!/usr/bin/env python3
"""
Representation geometry comparison for Visual Grounding models (Baseline vs +ITM).

DEFAULT: Fit a linear probe (logistic regression) on frozen z_reg to separate
positive vs negative pairs, and report probe-AUC.

Snapshot:
- Save everything needed to reproduce the plot later without ckpt/data.
- Replot from snapshot only.

Extract + plot + snapshot:
  python utils/plot_rep_geometry.py \
    --baseline artifacts/.../best.pth \
    --itm artifacts/.../best.pth \
    --backbone vit \
    --data data/lmdb/refcoco/val.lmdb \
    --num-samples 2000 \
    --out outputs/rep_geometry.pdf \
    --save-snapshot outputs/rep_geometry_snapshot.npz

Replot only (no ckpt/data needed):
  python utils/plot_rep_geometry.py \
    --load-snapshot outputs/rep_geometry_snapshot.npz \
    --out outputs/rep_geometry_replot.pdf
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import rcParams
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# UMAP
from umap import UMAP

# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from visual_grounding.config import ExperimentConfig
from visual_grounding.data import RefCOCODatasetLMDB, build_transforms
from visual_grounding.models import TransVG


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def set_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["text.usetex"] = False
    rcParams["mathtext.fontset"] = "stix"
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["figure.titlesize"] = 12
    rcParams["axes.titlesize"] = 10
    rcParams["axes.linewidth"] = 0.8
    rcParams["grid.alpha"] = 0.4


# -----------------------------------------------------------------------------
# AUC (Mann–Whitney U with tie handling)
# -----------------------------------------------------------------------------
def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def auc_pos_vs_neg(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    n_pos = pos_scores.size
    n_neg = neg_scores.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    ranks = _rankdata_average_ties(scores)

    r_pos_sum = ranks[:n_pos].sum()
    u = r_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_metrics_np(res: Dict[str, np.ndarray]) -> Dict[str, float]:
    auc = auc_pos_vs_neg(res["s_pos"], res["s_neg"])
    delta_mu = float(np.mean(res["s_pos"]) - np.mean(res["s_neg"]))
    return {"AUC": auc, "DeltaMu": delta_mu}


# -----------------------------------------------------------------------------
# Checkpoint loading (FIXED)
# -----------------------------------------------------------------------------
def _strip_prefix(k: str) -> str:
    for p in ("module.", "model.", "transvg.", "net."):
        if k.startswith(p):
            k = k[len(p) :]
    return k


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint state_dict is not a dict: {type(state)}")

    clean_state = {_strip_prefix(k): v for k, v in state.items()}

    try:
        model.load_state_dict(clean_state, strict=True)
        return
    except RuntimeError as e:
        print("[WARN] strict=True load failed. Trying strict=False.")
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        print(f"[WARN] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        if len(missing) > 50:
            raise RuntimeError(
                "Checkpoint/model likely mismatched backbone. "
                "Make sure --backbone matches the ckpt (vit/convnext/resnet50/clip-vit-b16)."
            ) from e


# -----------------------------------------------------------------------------
# CLIP-safe text sanitization (prevents CUDA gather OOB)
# -----------------------------------------------------------------------------
def sanitize_text_for_encoder(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.long()
    attention_mask = attention_mask.long()

    te = getattr(model, "text_encoder", None)
    cfg = getattr(te, "config", None) if te is not None else None

    max_len = getattr(cfg, "max_position_embeddings", None)
    vocab_size = getattr(cfg, "vocab_size", None)

    if max_len is not None and input_ids.dim() == 2 and input_ids.size(1) > max_len:
        input_ids = input_ids[:, :max_len].contiguous()
        attention_mask = attention_mask[:, :max_len].contiguous()

    if vocab_size is not None:
        bad = (input_ids < 0) | (input_ids >= vocab_size)
        if bad.any():
            input_ids = input_ids.clone()
            attention_mask = attention_mask.clone()
            input_ids[bad] = 0
            attention_mask[bad] = 0

    attention_mask = (attention_mask > 0).long()
    return input_ids, attention_mask


# -----------------------------------------------------------------------------
# Feature extraction: get z_reg only (probe will be trained later)
# -----------------------------------------------------------------------------
class FeatureExtractor:
    """
    Extract fused [REG] token representation (z_reg) from vl_transformer output.
    """

    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self._reg_features: Optional[torch.Tensor] = None

        if hasattr(model, "vl_transformer"):
            self.hook_handle = model.vl_transformer.register_forward_hook(self._hook_fn)
        else:
            raise ValueError("Could not find vl_transformer in model. Please check TransVG implementation.")

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self._reg_features = output[:, 0, :].detach().cpu()
        else:
            self._reg_features = output[0][:, 0, :].detach().cpu()

    @torch.no_grad()
    def extract_zreg(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        negative: bool = False,
    ) -> np.ndarray:
        img = images.to(self.device)

        txt = input_ids
        mask = attention_mask
        if negative:
            txt = torch.roll(txt, shifts=1, dims=0)
            mask = torch.roll(mask, shifts=1, dims=0)

        txt = txt.to(self.device)
        mask = mask.to(self.device)
        txt, mask = sanitize_text_for_encoder(self.model, txt, mask)

        _ = self.model(img, txt, mask, use_itm=False)

        if self._reg_features is None:
            raise RuntimeError("Hook did not capture REG features. Check hook point.")
        return self._reg_features.numpy()  # (B, D)

    def cleanup(self):
        self.hook_handle.remove()


def collect_features(
    ckpt_path: str,
    cfg: ExperimentConfig,
    backbone: str,
    loader: DataLoader,
    tag: str,
    device: str,
    limit: int,
) -> Dict[str, np.ndarray]:
    print(f"Loading {tag} from {ckpt_path} with backbone={backbone}")

    model = TransVG(backbone, cfg).to(device)
    model.eval()
    load_checkpoint_into_model(model, ckpt_path, device)

    extractor = FeatureExtractor(model, device)

    z_pos_list, z_neg_list = [], []
    count = 0
    pbar = tqdm(total=limit, desc=f"Extract z_reg ({tag})", leave=False)

    for batch in loader:
        if count >= limit:
            break

        imgs = batch["image"]
        txts = batch["input_ids"]
        masks = batch["attention_mask"]

        z_pos = extractor.extract_zreg(imgs, txts, masks, negative=False)
        z_neg = extractor.extract_zreg(imgs, txts, masks, negative=True)

        z_pos_list.append(z_pos)
        z_neg_list.append(z_neg)

        count += imgs.size(0)
        pbar.update(imgs.size(0))

    pbar.close()
    extractor.cleanup()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    Z_pos = np.concatenate(z_pos_list, axis=0)[:limit]
    Z_neg = np.concatenate(z_neg_list, axis=0)[:limit]
    return {"z_pos": Z_pos, "z_neg": Z_neg}


# -----------------------------------------------------------------------------
# Linear probe (DEFAULT)
# -----------------------------------------------------------------------------
@dataclass
class SplitIndices:
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_split_indices(n: int, seed: int, train_frac: float = 0.7) -> SplitIndices:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(train_frac * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:] if n_train < n else idx[:0]
    return SplitIndices(train_idx=train_idx.astype(np.int64), test_idx=test_idx.astype(np.int64))


def fit_linear_probe(
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    split: SplitIndices,
    *,
    seed: int,
    l2: float = 1e-4,
    lr: float = 1e-2,
    steps: int = 600,
) -> Dict[str, Any]:
    """
    Train logistic regression on frozen z_reg (pos=1, neg=0).
    Returns:
      - s_pos, s_neg: logits on TEST split (for AUC + hist)
      - probe params: w,b + standardization stats
    """
    z_pos = np.asarray(z_pos, dtype=np.float32)
    z_neg = np.asarray(z_neg, dtype=np.float32)
    n = min(len(z_pos), len(z_neg))
    z_pos = z_pos[:n]
    z_neg = z_neg[:n]
    d = z_pos.shape[1]

    train_idx = split.train_idx
    test_idx = split.test_idx

    if len(test_idx) == 0:
        # fallback: if n too small, just evaluate on train (not ideal but avoids crash)
        test_idx = train_idx

    X_tr = np.concatenate([z_pos[train_idx], z_neg[train_idx]], axis=0)  # (2*tr, D)
    y_tr = np.concatenate([np.ones(len(train_idx)), np.zeros(len(train_idx))], axis=0).astype(np.float32)

    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-6

    X_tr = (X_tr - mean) / std

    torch.manual_seed(seed)
    model = nn.Linear(d, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr).view(-1, 1)

    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()

    model.eval()

    def score(Z: np.ndarray) -> np.ndarray:
        Z = Z.astype(np.float32)
        Z = (Z - mean) / std
        with torch.no_grad():
            logits = model(torch.from_numpy(Z)).squeeze(1).cpu().numpy()
        return logits

    s_pos = score(z_pos[test_idx])
    s_neg = score(z_neg[test_idx])

    w = model.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])

    return {
        "s_pos": s_pos,
        "s_neg": s_neg,
        "probe_w": w,
        "probe_b": b,
        "x_mean": mean.reshape(-1),
        "x_std": std.reshape(-1),
        "test_idx": test_idx.astype(np.int64),
        "train_idx": train_idx.astype(np.int64),
    }


def attach_probe_scores(
    feats: Dict[str, np.ndarray],
    split: SplitIndices,
    *,
    seed: int,
) -> Dict[str, Any]:
    out = dict(feats)
    probe = fit_linear_probe(out["z_pos"], out["z_neg"], split, seed=seed)
    out["s_pos"] = probe["s_pos"]
    out["s_neg"] = probe["s_neg"]
    out["_probe"] = probe
    return out


def _align_lengths(
    base: Dict[str, Any], itm: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    n = min(len(base["z_pos"]), len(base["z_neg"]), len(itm["z_pos"]), len(itm["z_neg"]))
    base["z_pos"] = base["z_pos"][:n]
    base["z_neg"] = base["z_neg"][:n]
    itm["z_pos"] = itm["z_pos"][:n]
    itm["z_neg"] = itm["z_neg"][:n]
    return base, itm, n


# -----------------------------------------------------------------------------
# Snapshot IO
# -----------------------------------------------------------------------------
def save_snapshot(
    path: Path,
    *,
    baseline_res: Dict[str, Any],
    itm_res: Dict[str, Any],
    embedding: np.ndarray,
    base_metrics: Dict[str, float],
    itm_metrics: Dict[str, float],
    subset_indices: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)

    # store only what is needed to replot later
    np.savez_compressed(
        path,
        base_z_pos=baseline_res["z_pos"],
        base_z_neg=baseline_res["z_neg"],
        base_s_pos=baseline_res["s_pos"],
        base_s_neg=baseline_res["s_neg"],
        itm_z_pos=itm_res["z_pos"],
        itm_z_neg=itm_res["z_neg"],
        itm_s_pos=itm_res["s_pos"],
        itm_s_neg=itm_res["s_neg"],
        umap_embedding=embedding,
        base_auc=np.array([base_metrics["AUC"]], dtype=float),
        base_deltamu=np.array([base_metrics["DeltaMu"]], dtype=float),
        itm_auc=np.array([itm_metrics["AUC"]], dtype=float),
        itm_deltamu=np.array([itm_metrics["DeltaMu"]], dtype=float),
        subset_indices=subset_indices.astype(np.int64),
        meta_json=np.array([meta_json]),
    )
    print(f"[snapshot] Saved: {path}")


def load_snapshot(
    path: Path,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, float],
    Dict[str, float],
    np.ndarray,
    Dict[str, Any],
]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"][0]))

    baseline_res = {
        "z_pos": data["base_z_pos"],
        "z_neg": data["base_z_neg"],
        "s_pos": data["base_s_pos"],
        "s_neg": data["base_s_neg"],
    }
    itm_res = {
        "z_pos": data["itm_z_pos"],
        "z_neg": data["itm_z_neg"],
        "s_pos": data["itm_s_pos"],
        "s_neg": data["itm_s_neg"],
    }
    embedding = data["umap_embedding"]
    base_metrics = {"AUC": float(data["base_auc"][0]), "DeltaMu": float(data["base_deltamu"][0])}
    itm_metrics = {"AUC": float(data["itm_auc"][0]), "DeltaMu": float(data["itm_deltamu"][0])}
    subset_indices = data["subset_indices"].astype(np.int64)
    return baseline_res, itm_res, embedding, base_metrics, itm_metrics, subset_indices, meta


# -----------------------------------------------------------------------------
# Embedding + Plot
# -----------------------------------------------------------------------------
def compute_or_use_embedding(
    baseline_res: Dict[str, Any],
    itm_res: Dict[str, Any],
    *,
    embedding: Optional[np.ndarray],
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_seed: int,
) -> np.ndarray:
    if embedding is not None:
        return embedding

    print("Fitting UMAP...")
    X_all = np.concatenate(
        [baseline_res["z_pos"], baseline_res["z_neg"], itm_res["z_pos"], itm_res["z_neg"]],
        axis=0,
    )
    reducer = UMAP(
        metric=umap_metric,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_seed,
    )
    return reducer.fit_transform(X_all)


def plot_geometry(
    baseline_res: Dict[str, Any],
    itm_res: Dict[str, Any],
    save_path: str,
    *,
    embedding: Optional[np.ndarray] = None,
    base_metrics: Optional[Dict[str, float]] = None,
    itm_metrics: Optional[Dict[str, float]] = None,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    umap_seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    set_style()

    baseline_res, itm_res, N = _align_lengths(baseline_res, itm_res)

    embedding = compute_or_use_embedding(
        baseline_res,
        itm_res,
        embedding=embedding,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        umap_seed=umap_seed,
    )

    if embedding.shape[0] != 4 * N:
        raise ValueError(f"Embedding has {embedding.shape[0]} rows but expected 4*N={4*N} (N={N}).")

    base_pos_emb = embedding[0:N]
    base_neg_emb = embedding[N : 2 * N]
    itm_pos_emb = embedding[2 * N : 3 * N]
    itm_neg_emb = embedding[3 * N : 4 * N]

    if base_metrics is None:
        base_metrics = compute_metrics_np(baseline_res)
    if itm_metrics is None:
        itm_metrics = compute_metrics_np(itm_res)

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.9, 0.5])

    # (a)(b) UMAP
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    marker_size = 10
    alpha_pos = 0.45
    alpha_neg = 0.22

    all_x = embedding[:, 0]
    all_y = embedding[:, 1]
    xlim = (all_x.min() - 1, all_x.max() + 1)
    ylim = (all_y.min() - 1, all_y.max() + 1)

    for ax, pos, neg, title, color in zip(
        [ax1, ax2],
        [base_pos_emb, itm_pos_emb],
        [base_neg_emb, itm_neg_emb],
        ["(a) Baseline", "(b) Baseline + ITM"],
        ["#1f77b4", "#d62728"],
    ):
        ax.scatter(
            neg[:, 0],
            neg[:, 1],
            c="gray",
            marker="x",
            s=marker_size,
            alpha=alpha_neg,
            label="Negative Pair",
            linewidths=0.5,
        )
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=color,
            marker="o",
            s=marker_size,
            alpha=alpha_pos,
            label="Positive Pair",
            edgecolors="none",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    ax1.legend(loc="upper right", frameon=False, fontsize=8)

    # (c) probe score distributions
    ax3 = fig.add_subplot(gs[0, 2])
    bins = 30
    ax3.hist(baseline_res["s_pos"], bins=bins, density=True, histtype="step", color="#1f77b4", alpha=0.95, label="Base Pos", linewidth=1.5)
    ax3.hist(baseline_res["s_neg"], bins=bins, density=True, histtype="step", color="#1f77b4", linestyle="--", alpha=0.7, label="Base Neg")
    ax3.hist(itm_res["s_pos"], bins=bins, density=True, histtype="step", color="#d62728", alpha=0.95, label="ITM Pos", linewidth=1.5)
    ax3.hist(itm_res["s_neg"], bins=bins, density=True, histtype="step", color="#d62728", linestyle="--", alpha=0.7, label="ITM Neg")

    ax3.set_title("(c) Probe Score Dist.", fontweight="bold")
    ax3.set_xlabel("Linear Probe Logit (test split)")
    ax3.set_ylabel("Density")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=8, frameon=False)

    # (d) AUC bar
    ax4 = fig.add_subplot(gs[0, 3])
    x = np.arange(1)
    width = 0.35
    ax4.bar(x - width / 2, [base_metrics["AUC"]], width, label="Baseline", color="#1f77b4", alpha=0.85)
    ax4.bar(x + width / 2, [itm_metrics["AUC"]], width, label="ITM", color="#d62728", alpha=0.85)
    ax4.set_ylabel("AUC (Probe)")
    ax4.set_title("(d) Separability", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels([""])
    ax4.set_ylim(0.4, 1.0)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    print(f"Saving to {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("\n--- Probe Metrics (TEST split) ---")
    print(f"Baseline: AUC={base_metrics['AUC']:.3f}, Δμ={base_metrics['DeltaMu']:.3f}")
    print(f"ITM:      AUC={itm_metrics['AUC']:.3f}, Δμ={itm_metrics['DeltaMu']:.3f}")

    return embedding, base_metrics, itm_metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Snapshot mode
    parser.add_argument("--load-snapshot", type=str, default=None, help="Replot from .npz snapshot.")
    parser.add_argument("--save-snapshot", type=str, default=None, help="Save .npz snapshot after extraction.")

    # Extraction mode
    parser.add_argument("--baseline", type=str, default=None, help="Baseline checkpoint path.")
    parser.add_argument("--itm", type=str, default=None, help="ITM checkpoint path.")
    parser.add_argument("--backbone", type=str, default="vit", help="vit/convnext/resnet50/clip-vit-b16")
    parser.add_argument("--data", type=str, default=None, help="LMDB path, e.g. data/lmdb/refcoco/val.lmdb")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Probe config
    parser.add_argument("--probe-train-frac", type=float, default=0.7)
    parser.add_argument("--probe-steps", type=int, default=600)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-l2", type=float, default=1e-4)

    # UMAP
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")

    parser.add_argument("--out", type=str, default="outputs/rep_geometry.pdf")
    args = parser.parse_args()

    # Replay-only
    if args.load_snapshot is not None:
        base_res, itm_res, embedding, base_m, itm_m, subset_idx, meta = load_snapshot(Path(args.load_snapshot))
        print(f"[snapshot] Loaded: {args.load_snapshot}")
        print(f"[snapshot] Meta: {meta}")
        plot_geometry(base_res, itm_res, args.out, embedding=embedding, base_metrics=base_m, itm_metrics=itm_m)
        return

    if args.baseline is None or args.itm is None or args.data is None:
        raise SystemExit("Need --baseline, --itm, and --data unless using --load-snapshot.")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    cfg = ExperimentConfig()
    cfg.train.batch_size = 32
    cfg.data.num_workers = 4
    if str(args.backbone).startswith("clip"):
        # align with your eval script conventions if these exist
        if hasattr(cfg, "model") and hasattr(cfg.model, "text_encoder"):
            cfg.model.text_encoder = "openai/clip-vit-base-patch16"
        if hasattr(cfg, "data") and hasattr(cfg.data, "max_text_len"):
            cfg.data.max_text_len = 77

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    _, val_tfm = build_transforms(cfg.data.img_size)
    dataset = RefCOCODatasetLMDB(data_path, cfg, transform=val_tfm)

    # Fixed subset indices so baseline/itm see identical samples
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[: args.num_samples] if args.num_samples < len(dataset) else indices
    subset = Subset(dataset, subset_indices)

    loader = DataLoader(
        subset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Extract features
    feats_base = collect_features(args.baseline, cfg, args.backbone, loader, "Baseline", device, args.num_samples)
    feats_itm = collect_features(args.itm, cfg, args.backbone, loader, "ITM", device, args.num_samples)

    # Align lengths & make ONE shared split (fair)
    feats_base, feats_itm, N = _align_lengths(feats_base, feats_itm)
    split = make_split_indices(N, seed=args.seed, train_frac=args.probe_train_frac)

    # Attach probe scores (DEFAULT)
    res_base = attach_probe_scores(feats_base, split, seed=args.seed)
    res_itm = attach_probe_scores(feats_itm, split, seed=args.seed)

    # Plot
    embedding, base_metrics, itm_metrics = plot_geometry(
        res_base,
        res_itm,
        args.out,
        embedding=None,
        base_metrics=None,
        itm_metrics=None,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_seed=args.seed,
    )

    # Snapshot
    if args.save_snapshot is not None:
        meta = {
            "seed": int(args.seed),
            "backbone": str(args.backbone),
            "num_samples_requested": int(args.num_samples),
            "num_samples_effective": int(N),
            "data_path": str(args.data),
            "baseline_ckpt": str(args.baseline),
            "itm_ckpt": str(args.itm),
            "probe": {
                "train_frac": float(args.probe_train_frac),
                "steps": int(args.probe_steps),
                "lr": float(args.probe_lr),
                "l2": float(args.probe_l2),
                "train_idx_len": int(len(split.train_idx)),
                "test_idx_len": int(len(split.test_idx)),
            },
            "umap": {
                "metric": str(args.umap_metric),
                "n_neighbors": int(args.umap_neighbors),
                "min_dist": float(args.umap_min_dist),
                "random_state": int(args.seed),
            },
        }
        # store indices corresponding to effective N
        subset_idx_arr = np.array(subset_indices[:N], dtype=np.int64)

        save_snapshot(
            Path(args.save_snapshot),
            baseline_res=res_base,
            itm_res=res_itm,
            embedding=embedding,
            base_metrics=base_metrics,
            itm_metrics=itm_metrics,
            subset_indices=subset_idx_arr,
            meta=meta,
        )


if __name__ == "__main__":
    main()
