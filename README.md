# Capstone Project: RS-Adapted Open-Vocabulary Semantic Segmentation for Remote Sensing

> **Open-Vocabulary Semantic Segmentation for Remote Sensing Images**
> Built on [SegEarth-OV-3](https://arxiv.org/abs/2512.08730) with domain adaptation via Parameter-Efficient Fine-Tuning (PEFT) and integrated with [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) for interactive annotation.

---

## Overview

This project extends SegEarth-OV-3, a state-of-the-art Open-Vocabulary Semantic Segmentation (OVSS) model for remote sensing imagery. The original model uses SAM3's visual backbone with a built-in language backbone and a cross-modal transformer decoder for zero-shot grounding-based segmentation.

**Core contributions of this project:**

1. **RSAdapter + RSMultiscaleFPN** — Lightweight PEFT modules that inject remote sensing domain knowledge into SAM3's frozen ViT encoder and extract multi-scale features
2. **Feature Distillation Training** — Adds a Block 31 cosine distillation loss to preserve SAM3's visual-language alignment during domain adaptation, enabling OVSS performance to exceed the original baseline
3. **Category-Adaptive Dual-Head Fusion** — Replaces SegEarth-OV-3's fixed MAX fusion (Eq. 2) with category-aware adaptive weighting between the instance and semantic heads
4. **X-AnyLabeling Integration** — Wraps the full model stack into an interactive GUI annotation tool
5. **Concept Bank** *(planned)* — A structured visual-semantic concept bank for improved open-vocabulary generalization to unseen RS categories

---

## Repository Structure

```
capstone/
├── SegEarth-OV-3/
│   ├── rs_adapter/
│   │   ├── rs_adapter.py              # RSAdapter + RSMultiscaleFPN class definitions
│   │   ├── train_adapter.py           # Phase 1: closed-set PEFT training (CE + Dice)
│   │   └── train_adapter_distill.py   # Phase 6: distillation PEFT training (CE + Dice + Distill)
│   ├── eval_with_adapter.py           # OVSS evaluation with pre-trained adapter
│   ├── configs/cfg_loveda.py          # LoveDA eval config (7 classes, text prompts)
│   └── segearthov3_segmentor.py       # Original SegEarth-OV-3 OVSS segmentor
└── X-AnyLabeling/
    └── anylabeling/
        ├── services/auto_labeling/
        │   └── segearthov3.py         # SegEarth-OV-3 wrapper (Model subclass)
        └── configs/auto_labeling/
            └── segearthov3.yaml       # Model config (classes, thresholds, modes)
```

---

## Architecture

### Full Pipeline

```
Input Image (1008×1008)
    │
    ▼
SAM3 ViT Encoder (frozen, 840M params, depth=32, embed_dim=1024, patch_size=14)
    │   ┌──────────────────────────────────────────────────────┐
    │   │  RSAdapter ×32 (~4.2M, trained)                      │  ← PEFT: RS domain adaptation
    │   │  Bottleneck: Linear 1024→64 → GELU → Linear 64→1024  │
    │   │  + residual × scale  (scale initialized to 0)        │
    │   └──────────────────────────────────────────────────────┘
    │   Hook returns adapted output; all 32 subsequent blocks see adapted features
    │
    ├─── Block 7  output (global attention) ─────────────────────┐
    ├─── Block 15 output (global attention) ─────────────────────┤  RSMultiscaleFPN
    ├─── Block 23 output (global attention) ─────────────────────┤  (~3.4M, trained)
    └─── Block 31 output (global attention) ─────────────────────┘
                │
                ▼
        ┌──────────────────────────────────┐
        │ FPN Feature Pyramid              │
        │  p2: 288×288 (×4 bilinear up)   ← small objects
        │  p3: 144×144 (×2 bilinear up)   ← medium objects
        │  p4:  72×72  (native)            ← large objects
        │  p5:  36×36  (maxpool ÷2)       ← global context
        │  Top-down fusion: p5→p4→p3→p2 (3×3 Conv)
        └──────────────────────────────────┘
                │
    ┌───────────┴──────────────────────────────────────────────┐
    │ Closed-set training path                                 │  OVSS inference path
    │ FPN p4 → 1×1 Conv Head                                   │  Block 31 adapted →
    │ Loss: CE + 0.5×Dice (+ β×Distill in Phase 6)            │  SAM3 Neck → cross-modal decoder
    │                                                          │  → Dual-Head → Category-Adaptive Fusion
    └──────────────────────────────────────────────────────────┘
                │
                ▼
        Segmentation Map → Polygon Annotations (X-AnyLabeling)
```

> **OVSS inference path**: The FPN and 1×1 Conv Head are **not used** during OVSS evaluation. Only the RSAdapter hooks (modifying Block 31 output) affect the SAM3 Neck → cross-modal decoder pipeline.

### RSAdapter

Lightweight bottleneck adapter inserted at every ViT block output via `register_forward_hook`.

```
x (B, H, W, 1024) → flatten → Linear(1024→64) → GELU → Linear(64→1024) → × scale → + x
```

- `scale` initialized to `0`: identity transform at init, prevents disrupting pretrained features
- `adapted.to(x.dtype)`: handles bfloat16/float32 mixed precision
- **Parameters**: ~131K per block × 32 blocks = **~4.2M total** (~0.5% of SAM3)

### RSMultiscaleFPN

Fuses intermediate ViT global-attention block outputs (Blocks 7/15/23/31) via an FPN-style pyramid.
All four inputs are 72×72×1024 (global attention blocks output full-resolution in SAM3's ViT-Det).

```
f7  (72×72×1024) → Lateral Conv 1×1 (1024→256) → ×4 bilinear → 288×288  [p2: fine detail]
f15 (72×72×1024) → Lateral Conv 1×1 (1024→256) → ×2 bilinear → 144×144  [p3: medium]
f23 (72×72×1024) → Lateral Conv 1×1 (1024→256) →  identity   →  72×72   [p4: coarse]
f31 (72×72×1024) → Lateral Conv 1×1 (1024→256) → maxpool ÷2  →  36×36   [p5: global]

Top-down FPN fusion (3×3 Conv at each level):
  p5 → upsample → + p4 → upsample → + p3 → upsample → + p2
```

- **Parameters**: Lateral ×4 (~1.0M) + FPN Conv ×4 (~2.4M) = **~3.4M total**

### Feature Distillation (Phase 6)

Adds a cosine distillation constraint at Block 31 to prevent feature drift from degrading OVSS.

```
L_total = L_CE + 0.5 × L_Dice + β × L_distill

L_distill = 1 − cosine_sim(adapter(block_31_out), block_31_out.detach())
```

- **β = 0.1** (default): balances RS domain adaptation against SAM3 VL alignment preservation
- Hook stores `(adapted, original.detach())` pair for distillation while also writing `hook_feats` for the FPN path
- Target: Block 31 only (`--distill_blocks last`) — direct input to SAM3's Neck and cross-modal decoder

### Category-Adaptive Dual-Head Fusion

Replaces the fixed element-wise MAX fusion in SegEarth-OV-3 with per-category adaptive weighting.

**Baseline (MAX, original paper):**
```
P_fused(h,w) = max(P_sem(h,w), P_inst_agg(h,w))
```

**Heuristic fusion (implemented):**
```
P_fused^c(h,w) = α_c × P_inst_agg^c(h,w) + (1 − α_c) × P_sem^c(h,w)

α_c = 0.8  if c ∈ THINGS  (building, car, ship, …)
α_c = 0.2  if c ∈ STUFF   (road, water, farmland, …)
```

**Entropy-based dynamic fusion (implemented):**
```
H_inst = entropy(P_inst_agg_c)
H_sem  = entropy(P_sem_c)
α_c(h,w) = H_sem / (H_inst + H_sem)   # higher weight to the more confident head
```

Motivation: SegEarth-OV-3's own Table 3 shows "things" (buildings, vehicles) benefit more from the instance head while "stuff" (roads, water) benefit more from the semantic head — a distinction MAX fusion ignores.

---

## Experiment Results

### Dataset: LoveDA (Zenodo Official)

| Split | Urban | Rural | Total |
|-------|------:|------:|------:|
| Train | 1,156 | 1,366 | **2,522** |
| Val   |   765 |   904 | **1,669** |

Classes (7): `background` / `building` / `road` / `water` / `barren` / `forest` / `agricultural`

---

### Phase 1 — Closed-set Baseline (CE + Dice)

**Script**: `rs_adapter/train_adapter.py` | **Checkpoint**: `rs_adapter/ckpt_full_best.pt`

| Setting | Val mIoU |
|---------|:--------:|
| **Adapter + FPN (best)** | **0.5521** |

> Baseline / FPN-only / Adapter-only ablation not yet re-run on the Zenodo dataset.

**Per-class IoU (Adapter + FPN, best checkpoint):**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5462 | 0.6806 | 0.5879 | 0.7077 | 0.3248 | 0.4409 | 0.5765 |

---

### Phase 2 — OVSS Baseline (original SegEarth-OV-3, no adapter)

**Script**: `eval.py configs/cfg_loveda.py`

| mIoU | aAcc | mAcc |
|:----:|:----:|:----:|
| **47.38** | 63.80 | 62.01 |

**Per-class IoU:**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 45.53 | 63.80 | 53.87 | 51.40 | 35.79 | 33.81 | 47.46 |

> Barren (35.79) and Forest (33.81) are the weakest classes — SAM3's VL alignment has a larger text-visual gap for these vegetation-related categories.

---

### Phase 3 — OVSS + CE+Dice Adapter (Proxy Task Gap confirmed)

**Script**: `eval_with_adapter.py configs/cfg_loveda.py --adapter_ckpt rs_adapter/ckpt_full_best.pt`

| mIoU | vs Phase 2 |
|:----:|:----------:|
| **45.12** | **−2.26** |

**Per-class IoU (Δ vs Phase 2):**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 41.28 (−4.25) | **65.06 (+1.26)** | 53.52 (−0.35) | 49.54 (−1.86) | 28.82 (**−6.97**) | 28.60 (**−5.21**) | **49.00 (+1.54)** |

**Root cause — Proxy Task Gap:**

The CE+Dice loss optimizes pixel classification via the FPN path, but OVSS inference uses `Block 31 adapted → SAM3 Neck → cross-modal decoder` — a path the training loss never touched. The Adapter shifts Block 31 features toward FPN-classification-optimal representations, which disrupts the visual-text feature alignment the cross-modal decoder expects. Barren and Forest — already the weakest classes in Phase 2 — suffer the largest additional drops (−6.97, −5.21).

---

### Phase 6 — OVSS + Feature Distillation (Block 31)

**Script**: `rs_adapter/train_adapter_distill.py` → `eval_with_adapter.py configs/cfg_loveda.py --adapter_ckpt rs_adapter/ckpt_distill_best.pt`
**Checkpoint**: `rs_adapter/ckpt_distill_best.pt`

| Metric | Phase 2 (baseline) | Phase 6 (Distill, β=0.1) | Δ |
|--------|:-----------------:|:------------------------:|:---:|
| **OVSS mIoU** | 47.38 | **50.31** | **+2.94** |
| aAcc | 63.80 | 64.91 | +1.11 |
| mAcc | 62.01 | 66.72 | +4.72 |
| Closed-set val mIoU | — | 0.5490 | (Phase 1: 0.5521) |

**Per-class IoU (Δ vs Phase 2):**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 41.38 (−4.15) | **65.95 (+2.16)** | **55.68 (+1.80)** | **68.91 (+17.51)** | **40.23 (+4.48)** | 30.49 (−3.32) | **49.49 (+2.03)** |

5 of 7 classes exceed the Phase 2 baseline. Forest remains below baseline (30.49 vs 33.81); Background also declines (−4.15).

**Full Ablation Table (OVSS, Zenodo dataset):**

| Phase | Adapter | Loss | OVSS mIoU | barren IoU | forest IoU |
|-------|:-------:|------|:---------:|:----------:|:----------:|
| 2 (baseline) | ✗ | — | 47.38 | 35.79 | 33.81 |
| 3 | ✓ | CE + Dice | 45.12 | 28.82 | 28.60 |
| **6** | **✓** | **CE + Dice + Distill (β=0.1)** | **50.31** | **40.23** | **30.49** |
| (8) β sweep | ✓ | CE + Dice + Distill | ? | ? | ? |

---

## Training

### Common Hyperparameters

| Hyperparameter | Phase 1 | Phase 6 |
|---------------|:-------:|:-------:|
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Epochs | 20 | 20 |
| Batch Size | 2 | 4 |
| Mixed Precision | bfloat16 | bfloat16 |
| Seed | 42 | 42 |
| Resolution | 1008×1008 | 1008×1008 |

### Phase 1 — Closed-set (CE + Dice)

```bash
cd ~/capstone/SegEarth-OV-3
python rs_adapter/train_adapter.py \
    --segearthov3_path ~/capstone/SegEarth-OV-3 \
    --model_path       weights/sam3/sam3.pt \
    --data_root        ~/datasets/LoveDA_Final \
    --dataset          loveda \
    --bottleneck       64 \
    --lr               1e-3 \
    --epochs           20 \
    --batch_size       2 \
    --save_path        rs_adapter/ckpt_full.pt \
    --log_dir          runs/adapter
```

Ablation flags: `--no_adapter` (disable RSAdapter), `--no_fpn` (disable FPN)

### Phase 6 — Feature Distillation (CE + Dice + Distill)

```bash
cd ~/capstone/SegEarth-OV-3
python rs_adapter/train_adapter_distill.py \
    --segearthov3_path ~/capstone/SegEarth-OV-3 \
    --model_path       weights/sam3/sam3.pt \
    --data_root        ~/datasets/LoveDA_Final \
    --dataset          loveda \
    --distill_weight   0.1 \
    --distill_blocks   last \
    --epochs           20 \
    --batch_size       4 \
    --save_path        rs_adapter/ckpt_distill.pt \
    --log_dir          runs/adapter_distill
```

### OVSS Evaluation

```bash
cd ~/capstone/SegEarth-OV-3

# Phase 3: CE+Dice adapter
python eval_with_adapter.py configs/cfg_loveda.py \
    --adapter_ckpt rs_adapter/ckpt_full_best.pt

# Phase 6: Distillation adapter
python eval_with_adapter.py configs/cfg_loveda.py \
    --adapter_ckpt rs_adapter/ckpt_distill_best.pt
```

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.3+ with CUDA 12.1 (cu121 build)
- NVIDIA GPU with 16 GB+ VRAM (trained on NVIDIA A100-SXM4-80GB)

### Dataset (LoveDA, Zenodo)

Download from the [Zenodo official release](https://zenodo.org/record/5706578).

```
datasets/LoveDA_Final/
├── train/
│   ├── Urban/
│   │   ├── images_png/   # RGB .png (1024×1024), 1,156장
│   │   └── masks_png/    # Label .png (1–7, 0=no-data ignored)
│   └── Rural/
│       ├── images_png/   # 1,366장
│       └── masks_png/
└── validation/
    ├── Urban/
    │   ├── images_png/   # 765장
    │   └── masks_png/
    └── Rural/
        ├── images_png/   # 904장
        └── masks_png/
```

Label mapping: `1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agricultural`

---

## X-AnyLabeling Integration

### Config (`segearthov3.yaml`)

```yaml
classes:
  - "background"
  - "building"
  - "road,highway"
  - "water,river,lake"
  - "barren,bareland"
  - "forest,tree"
  - "agricultural,farmland,cropland"

prob_thd: 0.3            # Final pixel-level filter (also controlled via UI slider)
fusion_mode: "entropy"   # "max" | "heuristic" | "entropy"
adapter_mode: "full"     # "off" | "full"
adapter_path: "/path/to/ckpt_distill_best.pt"
adapter_bottleneck: 64
```

### Fusion Modes

| `fusion_mode` | Description |
|---|---|
| `max` | Element-wise MAX (original SegEarth-OV-3) |
| `heuristic` | Fixed α per things/stuff category |
| `entropy` | Dynamic pixel-wise α from prediction entropy |

### Adapter Modes

| `adapter_mode` | Description |
|---|---|
| `off` | Original SegEarth-OV-3 (no adapter) |
| `full` | Load pre-trained RSAdapter + FPN weights |

---

## Planned: Concept Bank

> **Status: Future Implementation**

SegEarth-OV-3 relies on manually curated text prompts (e.g., `"tree,forest"`, `"water,river"`) with no systematic exploration of prompt design. The paper itself acknowledges: *"The prompt setting is manually curated based on dataset features and has not been systematically explored in this version."*

The planned **Concept Bank** addresses this by building a structured repository of visual-semantic concepts for RS categories.

**Planned scope:**

- **Hierarchical prompt generation**: Map coarse RS categories → fine-grained sub-concepts (e.g., `building` → `residential / commercial / industrial`), then use SAM3 presence score to automatically select the most relevant sub-prompt per image
- **Automatic synonym expansion**: Use domain-specific ontologies (WordNet, GeoNames) to auto-generate synonyms, ensemble predictions across synonym variants
- **Concept-level presence scoring**: Per-patch presence score instead of global scalar to suppress false positives in multi-patch inference

This component requires no additional training — it operates purely at inference time as a training-free complement to the RSAdapter PEFT approach.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `sam3_model.backbone.forward_image()` called directly | `Sam3Processor.set_image()` has `@inference_mode()` — blocks gradients during training |
| `hook_feats` stored without `.detach()` | Gradient must flow through FPN → Adapter during closed-set training |
| OVSS eval hook: `return adapter(output)` only | FPN not in OVSS path; only Block 31 adapted output feeds SAM3 Neck |
| `scale = zeros(1)` init in RSAdapter | Identity at start; prevents catastrophic forgetting early in PEFT |
| `original.detach()` in distillation hook | Prevents gradient flowing into the frozen SAM3 teacher path |
| Distillation target: Block 31 only | Direct input to SAM3 Neck — most causally linked to OVSS performance |
| β = 0.1 for distillation weight | Empirically balances RS domain adaptation vs VL alignment preservation |
| CE + 0.5 × Soft Dice | CE handles class imbalance; Dice improves boundary IoU |

---

## Roadmap

- [x] RSAdapter + RSMultiscaleFPN implementation (`rs_adapter/rs_adapter.py`)
- [x] Phase 1 closed-set training (CE + Dice): Adapter+FPN best val mIoU **0.5521**
- [x] Phase 2 OVSS baseline: mIoU **47.38**
- [x] Phase 3 OVSS + CE+Dice Adapter: mIoU **45.12** (−2.26, Proxy Task Gap confirmed)
- [x] Phase 6 Feature Distillation: OVSS mIoU **50.31** (+2.94 vs baseline)
- [x] Category-Adaptive Fusion (heuristic + entropy)
- [x] X-AnyLabeling integration
- [ ] Phase 1 Ablation re-run (Baseline / FPN-only / Adapter-only on Zenodo dataset)
- [ ] Phase 8: β sweep ablation {0.01, 0.05, 0.1, 0.5}
- [ ] Full ablation: Fusion × Adapter × OVSS
- [ ] Concept Bank: hierarchical prompt + synonym expansion (inference-time, no retraining)

---

## References

- SegEarth-OV-3: [earth-insights/SegEarth-OV-3](https://github.com/earth-insights/SegEarth-OV-3)
- X-AnyLabeling: [CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- LoveDA Dataset: [LoveDA Zenodo](https://zenodo.org/record/5706578) / [GitHub](https://github.com/Junjue-Wang/LoveDA)
- SAM3: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
