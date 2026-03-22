## 1. Architecture Overview
The model follows a transformer-based encoder-decoder paradigm for one-stage visual grounding:

1.  **Visual Encoder:** Extracts dense visual features. Feature maps are flattened (or already flat) and projected.
    *   **ViT (Vision Transformer):** Pure transformer backbone (`vit_base_patch16_224`).
    *   **ConvNeXt:** Modern large-kernel CNN (`convnextv2_base`).
    *   **CLIP (ViT-B/16):** Pre-trained vision transformer from CLIP (OpenAI).
    *   **ResNet:** Traditional CNN baseline (`resnet50`, `resnet101`).

2.  **Linguistic Encoder:** Processes the referring expression.
    *   **BERT:** Standard BERT-base encoder.
    *   **CLIP Text:** Used when the visual backbone is CLIP, ensuring feature alignment.

3.  **Visual-Linguistic Fusion:**
    A multimodal transformer fuses visual tokens, text tokens, and a learnable `[REG]` token. The `[REG]` token aggregates context for prediction.

4.  **Prediction Heads:**
    *   **Coordinate Regression:** MLP predicting `(x, y, w, h)` coordinates. Optimizes **L1** and **GIoU** loss.
    *   **Image-Text Matching (ITM):** **(New)** An auxiliary binary classification head branching off the `[REG]` token. It verifies whether the predicted region corresponds to the text, optimizing **Binary Cross-Entropy (BCE)** loss.

## 2. Environment Setup

This project uses `uv` for fast dependency management.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies (creates .venv automatically)
uv sync

# 3. Activate environment
source .venv/bin/activate
```

## 3. Data Preparation

The framework expects **RefCOCO** datasets pre-packaged into LMDB format for high-throughput training.

Structure your data directory as follows:
```
data/
└── lmdb/
    └── refcoco/
        ├── train.lmdb
        └── val.lmdb
```

## 4. Training

**Using the CLIP Backbone:**
CLIP requires specific handling. Note that `--backbones` accepts `clip` or `clip-vit-b16`.
```bash
python train.py \
  --backbones clip \
  --clip-model-name openai/clip-vit-base-patch16 \
  --num-epochs 50 \
  --batch-size 64
```

**Initializing CLIP from FineCLIP (custom checkpoint):**
If you have a local `.pt` file such as `checkpoints/FineCLIP_coco_vitb16.pt`, load it directly via the new CLI flags.
```bash
python train.py \
  --backbones clip \
  --clip-model-name openai/clip-vit-base-patch16 \
  --clip-checkpoint checkpoints/FineCLIP_coco_vitb16.pt \
  --use-itm \
  --itm-weight 0.5 \
  --num-epochs 50 \
  --batch-size 64
```

**Initializing CLIP from HyCoCLIP (ViT-B checkpoint):**
Download the checkpoint and pass it through the same `--clip-checkpoint` argument.
```bash
huggingface-cli download avik-pal/hycoclip hycoclip_vit_b.pth --local-dir checkpoints

python train.py \
  --backbones clip \
  --clip-model-name openai/clip-vit-base-patch16 \
  --clip-checkpoint checkpoints/hycoclip_vit_b.pth \
  --num-epochs 50 \
  --batch-size 64
```
*Tip: `checkpoints/clip_vit_b.pth` from the same repository is also supported.*

**Resuming Training:**
Resume from a specific checkpoint. This restores model weights, optimizer state, and scheduler.
```bash
python train.py \
  --backbones clip \
  --clip-model-name openai/clip-vit-base-patch16 \
  --clip-checkpoint checkpoints/FineCLIP_coco_vitb16.pt \
  --resume artifacts/ckpt_clip_itm_0.0_260316_0745/checkpoints/last.pth \
  --num-epochs 50 \
  --use-cosine-scheduler
```
*Note: Set `--num-epochs` to the total target epochs (e.g., if resuming at epoch 40/100, set 100).*

## 5. Inference and Evaluation

### RefCOCO test split evaluation (LMDB)

Use `test.py` to evaluate a trained checkpoint on RefCOCO LMDB splits.

```bash
# testA
python test.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --split testB \
  --no-visualize

# testB
python test.py clip \
  -c artifacts/ckpt_clip_itm_0.0_260316_1136/checkpoints/best.pth \
  --split testB \
  --no-visualize
```

### Cross-dataset generalization on RefCOCOg (parquet)

Use the zero-shot evaluator to test transfer from RefCOCO-trained checkpoints to RefCOCOg.

```bash
# val
python eval/zero_shot_refcocog.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --dataset refcocog \
  --split val

# test
python eval/zero_shot_refcocog.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --dataset refcocog \
  --split test
```

### Cross-dataset generalization on RefCOCO+ (parquet)

The same evaluator supports RefCOCO+ parquet shards.

```bash
# val
python eval/zero_shot_refcocog.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --dataset refcocoplus \
  --split val

# testA
python eval/zero_shot_refcocog.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --dataset refcocoplus \
  --split testA

# testB
python eval/zero_shot_refcocog.py clip \
  -c artifacts/meru_260316_1643/checkpoints/best.pth \
  --dataset refcocoplus \
  --split testB
```

If the checkpoint was trained with ITM enabled, append `--use-itm` to the commands above.

### CLI Reference

| Argument | Description | Default |
|---|---|---|
| `--backbones` | List of backbones: `vit`, `convnext`, `resnet50`, `clip-vit-b16` | `vit convnext` |
| `--num-epochs` | Total training epochs | `20` |
| `--batch-size` | Logical batch size | `64` |
| `--use-cosine-scheduler` | Enable Cosine Annealing LR scheduler | `False` |
| `--use-compile` | Enable `torch.compile` (requires PyTorch 2.0+) | `True` |
| `--clip-model-name` | Hugging Face id / local folder for CLIP backbone | `openai/clip-vit-base-patch16` |
| `--clip-checkpoint` | Path to a `.pt` checkpoint (e.g., FineCLIP) loaded into CLIP weights | `None` |

python test.py clip -c artifacts/meru_260316_1643/checkpoints/best.pth --split testB --vis-samples 20 --vis-dir outputs/predictions/fineclip_260316_1643_testB --vis-dpi 300 --vis-seed 41# clip_vg_rep
