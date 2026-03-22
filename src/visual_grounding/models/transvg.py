"""TransVG implementation with interchangeable backbones."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, CLIPModel

from ..config import ExperimentConfig


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron used for the bbox regressor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class VisualLinguisticTransformer(nn.Module):
    """Cross-modal encoder used by TransVG."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.transformer(tokens, src_key_padding_mask=mask)
        return self.norm(out)


class TransVG(nn.Module):
    """TransVG model that supports both ViT and ConvNeXt backbones."""

    def __init__(self, backbone_type: str, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone_type = backbone_type.lower()
        self.vision_encoder_type = "timm"

        if self.backbone_type == "vit":
            self.vision_encoder = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
            vision_dim = 768
            self.num_vis_tokens = 196
        elif self.backbone_type == "convnext":
            self.vision_encoder = timm.create_model(
                "convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0
            )
            vision_dim = 1024
            self.num_vis_tokens = 49
        elif self.backbone_type in {"clip", "clip-vit-b16"}:
            clip_model_name = getattr(cfg.model, "clip_model_name", "openai/clip-vit-base-patch16")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            clip_checkpoint_path = getattr(cfg.model, "clip_checkpoint_path", None)
           
            if clip_checkpoint_path:
                self._load_clip_checkpoint(Path(clip_checkpoint_path))
            self.vision_encoder = self.clip_model.vision_model
            self.text_encoder = self.clip_model.text_model
            self.vision_encoder_type = "clip"

            vision_dim = self.clip_model.config.vision_config.hidden_size
            text_dim = self.clip_model.config.text_config.hidden_size
            patch = self.clip_model.config.vision_config.patch_size
            image_size = self.clip_model.config.vision_config.image_size
            self.num_vis_tokens = (image_size // patch) ** 2
            self.num_text_tokens = self.clip_model.config.text_config.max_position_embeddings
        elif self.backbone_type == "resnet50":
            self.vision_encoder = timm.create_model(
                "resnet50", pretrained=True, num_classes=0
            )
            vision_dim = 2048
            self.num_vis_tokens = 49
        elif self.backbone_type == "resnet101":
            self.vision_encoder = timm.create_model(
                "resnet101", pretrained=True, num_classes=0
            )
            vision_dim = 2048
            self.num_vis_tokens = 49
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        if self.backbone_type not in {"clip", "clip-vit-b16"}:
            self.text_encoder = BertModel.from_pretrained(cfg.model.text_encoder)
            text_dim = 768
            self.num_text_tokens = cfg.data.max_text_len

        self.vision_proj = nn.Linear(vision_dim, cfg.model.hidden_dim)
        self.text_proj = nn.Linear(text_dim, cfg.model.hidden_dim)

        self.reg_token = nn.Embedding(1, cfg.model.hidden_dim)
        total_tokens = 1 + self.num_vis_tokens + self.num_text_tokens
        self.vl_pos_embed = nn.Embedding(total_tokens, cfg.model.hidden_dim)

        self.vl_transformer = VisualLinguisticTransformer(
            d_model=cfg.model.hidden_dim,
            nhead=cfg.model.vl_num_heads,
            num_layers=cfg.model.vl_num_layers,
            dropout=0.1,
        )
        self.bbox_head = MLP(cfg.model.hidden_dim, cfg.model.hidden_dim, 4, 3)

        # Image-text matching head (auxiliary task)
        self.itm_head = nn.Sequential(
            nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.hidden_dim, 1),
        )

    def _encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """Extract and project visual tokens from the backbone."""
        if self.backbone_type == "vit":
            features = self.vision_encoder.forward_features(image)
            if isinstance(features, dict):
                features = features["x"]
            visual_feat = features[:, 1:, :]
        elif self.vision_encoder_type == "clip":
            outputs = self.vision_encoder(pixel_values=image)
            visual_feat = outputs.last_hidden_state[:, 1:, :]
        else:
            features = self.vision_encoder.forward_features(image)
            if isinstance(features, dict):
                features = features["x"]
            b, c, h, w = features.shape
            visual_feat = features.flatten(2).transpose(1, 2)
        return self.vision_proj(visual_feat)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract and project text tokens from BERT."""
        if self.backbone_type in {"clip", "clip-vit-b16"}:
            text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(text_output.last_hidden_state)

    def _load_clip_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a custom CLIP checkpoint (e.g., FineCLIP) for initialization."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state", "model", "module", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        if not isinstance(state_dict, dict):
            raise ValueError("Loaded CLIP checkpoint is not a valid state dict")

        def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            prefixes = (
                "module.",
                "model.",
                "clip_model.",
                "clip.",
                "student.",
            )
            out: Dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                key = k
                changed = True
                while changed:
                    changed = False
                    for p in prefixes:
                        if key.startswith(p):
                            key = key[len(p):]
                            changed = True
                            break
                out[key] = v
            return out

        def _loaded_count(incompatible_obj, total_keys: int) -> int:
            return max(total_keys - len(incompatible_obj.unexpected_keys), 0)

        state_dict = _strip_prefixes(state_dict)

        clip_keys = set(self.clip_model.state_dict().keys())
        loaded_main = sum(1 for k in state_dict.keys() if k in clip_keys)
        if loaded_main > 0:
            incompatible = self.clip_model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded %d/%d parameters into CLIPModel from %s", loaded_main, len(state_dict), checkpoint_path)
            if incompatible.missing_keys:
                logger.warning("Missing keys when loading CLIP checkpoint: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys when loading CLIP checkpoint: %s", incompatible.unexpected_keys)
            return

        # Fallback for checkpoints that store component weights only.
        vision_candidates: Dict[str, torch.Tensor] = {}
        text_candidates: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("vision_model."):
                vision_candidates[k.replace("vision_model.", "", 1)] = v
            elif k.startswith("visual."):
                vision_candidates[k.replace("visual.", "", 1)] = v
            elif k.startswith("text_model."):
                text_candidates[k.replace("text_model.", "", 1)] = v

        vision_loaded = 0
        text_loaded = 0
        if vision_candidates:
            inc_v = self.vision_encoder.load_state_dict(vision_candidates, strict=False)
            vision_loaded = _loaded_count(inc_v, len(vision_candidates))
        if text_candidates:
            inc_t = self.text_encoder.load_state_dict(text_candidates, strict=False)
            text_loaded = _loaded_count(inc_t, len(text_candidates))

        if vision_loaded == 0 and text_loaded == 0:
            logger.warning(
                "Could not map CLIP checkpoint keys automatically. Keeping default pretrained CLIP weights from %s",
                self.cfg.model.clip_model_name,
            )
            logger.warning("First 20 checkpoint keys: %s", list(state_dict.keys())[:20])
            return

        logger.info(
            "Loaded fallback CLIP components from %s (vision: %d/%d, text: %d/%d)",
            checkpoint_path,
            vision_loaded,
            len(vision_candidates),
            text_loaded,
            len(text_candidates),
        )

    def _fuse(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Fuse visual and text tokens through the VL transformer and return the [REG] token."""
        batch_size = visual_tokens.size(0)
        device = visual_tokens.device

        reg_tokens = self.reg_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        all_tokens = torch.cat([reg_tokens, visual_tokens, text_tokens], dim=1)

        pos_embed = self.vl_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        all_tokens = all_tokens + pos_embed

        reg_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        visual_mask = torch.zeros(batch_size, self.num_vis_tokens, dtype=torch.bool, device=device)
        text_mask = ~attention_mask.bool()
        combined_mask = torch.cat([reg_mask, visual_mask, text_mask], dim=1)

        fused_tokens = self.vl_transformer(all_tokens, mask=combined_mask)
        return fused_tokens[:, 0, :]

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_itm: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        visual_tokens = self._encode_vision(image)
        text_tokens = self._encode_text(input_ids, attention_mask)

        # ── Positive pair (matched image-text) ──────────────────────
        reg_output = self._fuse(visual_tokens, text_tokens, attention_mask)
        bbox_pred = self.bbox_head(reg_output).sigmoid()

        if not use_itm:
            return bbox_pred

        # ── ITM branch ──────────────────────────────────────────────
        batch_size = image.size(0)
        pos_logits = self.itm_head(reg_output).squeeze(-1)  # (B,)

        # Create negative pairs by shifting text by 1 within the batch
        shift = 1
        neg_text_tokens = torch.roll(text_tokens, shifts=shift, dims=0)
        neg_attention_mask = torch.roll(attention_mask, shifts=shift, dims=0)
        neg_reg_output = self._fuse(visual_tokens, neg_text_tokens, neg_attention_mask)
        neg_logits = self.itm_head(neg_reg_output).squeeze(-1)  # (B,)

        # (2B,) logits and labels
        itm_logits = torch.cat([pos_logits, neg_logits], dim=0)
        itm_labels = torch.cat([
            torch.ones(batch_size, device=image.device),
            torch.zeros(batch_size, device=image.device),
        ], dim=0)

        return {
            "bbox_pred": bbox_pred,
            "itm_logits": itm_logits,
            "itm_labels": itm_labels,
        }


__all__ = ["TransVG", "VisualLinguisticTransformer", "MLP"]
