# Capstone Project: RS-Adapted Open-Vocabulary Semantic Segmentation for Remote Sensing

> **Open-Vocabulary Semantic Segmentation for Remote Sensing Images**
> Built on [SegEarth-OV-3](https://arxiv.org/abs/2512.08730) with domain adaptation via Parameter-Efficient Fine-Tuning (PEFT) and integrated with [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) for interactive annotation.

---

## Overview

This project extends SegEarth-OV-3, a state-of-the-art Open-Vocabulary Semantic Segmentation (OVSS) model for remote sensing imagery. The original model uses a frozen SAM3 visual encoder with CLIP-based text-image similarity for zero-shot segmentation.

**Core contributions of this project:**

1. **RSAdapter + RSMultiscaleFPN** — Lightweight PEFT modules that inject remote sensing domain knowledge into SAM3's frozen ViT encoder and extract multi-scale features for small-object segmentation
2. **Category-Adaptive Dual-Head Fusion** — Replaces SegEarth-OV-3's fixed MAX fusion (Eq. 2) with category-aware adaptive weighting between the instance and semantic heads
3. **X-AnyLabeling Integration** — Wraps the full model stack (SAM3 + Adapter + FPN + Fusion) into an interactive GUI annotation tool
4. **Concept Bank** *(planned)* — A structured visual-semantic concept bank for improved open-vocabulary generalization to unseen RS categories

---

## Repository Structure

```
capstone/
├── SegEarth-OV-3/
│   ├── rs_adapter/
│   │   ├── rs_adapter.py          # RSAdapter + RSMultiscaleFPN class definitions
│   │   └── train_adapter.py       # PEFT training script (LoveDA / iSAID)
│   ├── sam3/                      # SAM3 model core
│   ├── weights/sam3/sam3.pt       # Pre-trained SAM3 weights (3.3 GB, not tracked)
│   ├── segearthov3_segmentor.py   # Original SegEarth-OV-3 OVSS evaluator
│   └── test_fusion_modes.py       # Fusion mode validation script
└── X-AnyLabeling/
    └── anylabeling/
        ├── services/auto_labeling/
        │   └── segearthov3.py     # SegEarth-OV-3 wrapper (Model subclass)
        └── configs/auto_labeling/
            └── segearthov3.yaml   # Model config (classes, thresholds, modes)
```

---

## Architecture

### Full Pipeline

![Adapter + FPN Architecture]('revised Architecture.png')

> **Multi-scale Adapter with FPN Fusion:** Lightweight bottleneck adapters (~0.5% of SAM3's 840M parameters) are inserted into all 32 ViT blocks via forward hooks. Intermediate features from global-attention checkpoints (Blocks 7, 15, 23, 31) are aggregated through an FPN-style top-down pathway to produce a four-level feature pyramid (P2–P5), injecting RS domain knowledge while preserving pretrained representations.

```
Input Image
    │
    ▼
SAM3 ViT Encoder (frozen, 840M params)
    │   ┌─────────────────────────────────┐
    │   │  RSAdapter ×32 (~4.2M, trained) │  ← PEFT: RS domain knowledge injection
    │   │  (bottleneck: Linear 1024→64→1024, scale=0 init)
    │   └─────────────────────────────────┘
    │
    ├─── Block 7  output (f7)  ─────────────┐
    ├─── Block 15 output (f15) ─────────────┤
    ├─── Block 23 output (f23) ─────────────┤  RSMultiscaleFPN (~3.4M, trained)
    └─── Block 31 output (f31) ─────────────┘
                │
                ▼
        ┌───────────────────┐
        │  FPN Feature Pyramid        │
        │  p2: 288×288 (×4 up)  ← small objects
        │  p3: 144×144 (×2 up)  ← medium objects
        │  p4:  72×72  (native) ← large objects
        │  p5:  36×36  (pool)   ← global context
        └───────────────────┘
                │
    ┌───────────┴───────────────────────────────┐
    │ Closed-set (PEFT training)                │  Open-Vocabulary (OVSS inference)
    │ 1×1 Conv Head → logits                   │  SAM3 Dual-Head × text embeddings
    │                                           │  → Category-Adaptive Fusion
    └───────────────────────────────────────────┘
                │
                ▼
        Segmentation Map → Polygon Annotations (X-AnyLabeling)
```

### RSAdapter

A lightweight bottleneck adapter inserted at every ViT transformer block output.

```
x → Linear(1024→64) → GELU → Linear(64→1024) → × scale → + x
```

- `scale` initialized to `0`: identity at start, enabling stable PEFT training
- `adapted.to(x.dtype)`: handles bfloat16/float32 mixed precision
- **Parameters**: ~131K per block × 32 blocks = **~4.2M total** (~0.5% of SAM3)

### RSMultiscaleFPN

Fuses intermediate ViT block outputs (f7, f15, f23, f31) via an FPN-style pyramid.

```
f7  (72×72×1024) → Lateral Conv 1×1 → ×4 upsample → 288×288  [p2: fine detail]
f15 (72×72×1024) → Lateral Conv 1×1 → ×2 upsample → 144×144  [p3: medium]
f23 (72×72×1024) → Lateral Conv 1×1 →   identity  →  72×72   [p4: coarse]
f31 (72×72×1024) → Lateral Conv 1×1 → maxpool ÷2  →  36×36   [p5: global]

Top-down FPN fusion (3×3 Conv):
  p5 → upsample → + p4 → upsample → + p3 → upsample → + p2
```

- **Parameters**: Lateral ×4 (~1.0M) + FPN Conv ×4 (~2.4M) = **~3.4M total**

### Category-Adaptive Dual-Head Fusion

Replaces the fixed element-wise MAX fusion in SegEarth-OV-3 (Eq. 2) with per-category adaptive weighting.

**Baseline (MAX, original paper):**
```
P_fused(h,w) = max(P_sem(h,w), P_inst_agg(h,w))
```

**Heuristic fusion (Phase 1, implemented):**
```
P_fused^c(h,w) = α_c × P_inst_agg^c(h,w) + (1 - α_c) × P_sem^c(h,w)

α_c = 0.8  if c ∈ THINGS  (building, car, ship, …)
α_c = 0.2  if c ∈ STUFF   (road, water, farmland, …)
```

**Entropy-based dynamic fusion (Phase 2, implemented):**
```
H_inst = entropy(P_inst_agg_c)    # instance head uncertainty
H_sem  = entropy(P_sem_c)         # semantic head uncertainty

α_c(h,w) = H_sem / (H_inst + H_sem)   # higher weight to more confident head
```

Motivation: SegEarth-OV-3's own Table 3 shows that "things" categories (buildings, vehicles) benefit more from the instance head while "stuff" categories (roads, water) benefit more from the semantic head — a distinction that MAX fusion ignores.

---

## Implemented Novelties

| # | Contribution | Status | Location |
|---|---|---|---|
| 1 | RSAdapter (PEFT domain adaptation) | ✅ Done | `rs_adapter/rs_adapter.py` |
| 2 | RSMultiscaleFPN (multi-scale features) | ✅ Done | `rs_adapter/rs_adapter.py` |
| 3 | PEFT training pipeline (LoveDA / iSAID) | ✅ Done | `rs_adapter/train_adapter.py` |
| 4 | Category-Adaptive Fusion (heuristic) | ✅ Done | `segearthov3.py` L308–387 |
| 5 | Category-Adaptive Fusion (entropy-based) | ✅ Done | `segearthov3.py` L308–387 |
| 6 | X-AnyLabeling integration | ✅ Done | `segearthov3.py` |
| 7 | Concept Bank for OVSS generalization | 🔧 Planned | — |

---

## Planned: Concept Bank

> **Status: Future Implementation**

SegEarth-OV-3 relies on manually curated text prompts (e.g., `"tree,forest"`, `"water,river"`) with no systematic exploration of prompt design. The planned **Concept Bank** addresses this limitation by building a structured repository of visual-semantic concepts for RS categories.

**Planned scope:**
- A hierarchical concept bank mapping coarse RS categories → fine-grained sub-concepts (e.g., `building` → `residential / commercial / industrial`)
- Automatic synonym expansion using domain-specific ontologies (WordNet, GeoNames)
- Concept-level presence scoring: for each patch, rank sub-concepts by SAM3 presence score and select the most relevant prompt automatically
- Ensemble of concept-level predictions for improved zero-shot robustness

This component will extend the OVSS inference pipeline end-to-end without requiring additional training, positioning it as a training-free complement to the RSAdapter PEFT approach.

---

## Experiment Plan & Results

### Phase 1 — Closed-set Ablation (LoveDA, 20 epochs)

Evaluates visual feature quality of SAM3 + Adapter + FPN via supervised segmentation on LoveDA (7 classes).

**Ablation configuration:**

| Exp | Adapter | FPN | Flags | Trainable Params |
|-----|:-------:|:---:|-------|-----------------|
| Baseline | ✗ | ✗ | `--no_adapter --no_fpn` | Head (~0.002M) |
| +FPN only | ✗ | ✓ | `--no_adapter` | FPN + Head (~3.4M) |
| +Adapter only | ✓ | ✗ | `--no_fpn` | Adapter + Head (~4.2M) |
| +FPN+Adapter | ✓ | ✓ | (default) | Adapter + FPN + Head (~7.6M) |

**Results (Val mIoU):**

| | With Adapter | Without Adapter (`--no_adapter`) |
|---|:---:|:---:|
| FPN | **0.6384** | 0.6162 |
| No FPN (`--no_fpn`) | 0.6324 | 0.4905 |

**Summary:**

| Exp | Adapter | FPN | Params | Val mIoU |
|-----|:-------:|:---:|-------:|:--------:|
| Baseline | ✗ | ✗ | ~0.002M | 0.4905 |
| +FPN only | ✗ | ✓ | ~3.4M | 0.6162 |
| +Adapter only | ✓ | ✗ | ~4.2M | 0.6324 |
| **+FPN+Adapter** | ✓ | ✓ | **~7.6M** | **0.6384** |

> FPN contributes the largest single gain (+0.126 mIoU over Baseline). RSAdapter adds an additional +0.022 on top of FPN. Without FPN, the adapter alone still improves significantly over Baseline (+0.142).

**Class-wise Val IoU — +FPN+Adapter best checkpoint (Epoch 7):**

| Background | Building | Road | Water | Barren | Forest | Agricultural |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.9940 | 0.5171 | 0.6669 | 0.5222 | 0.7277 | 0.4108 | 0.6304 |

> Forest IoU (0.411) remains the lowest, likely due to spectral confusion with Agricultural. Best checkpoint was reached at Epoch 7/20, suggesting early convergence.

### Phase 2 — OVSS Baseline (planned)

Run original SegEarth-OV-3 (no adapter, text-prompt based) on LoveDA val split.
Provides ① baseline mIoU for OVSS comparison.

### Phase 3 — OVSS + Adapter (planned)

Load Phase 1 adapter weights, plug into OVSS inference (text prompt + max/argmax).
Measures whether RS domain adaptation generalizes to the open-vocabulary setting.

### Phase 4 — Full Ablation: Fusion × Adapter (planned)

| | `adapter: off` | `adapter: full` |
|---|---|---|
| `fusion: max` | ① Baseline | ② + Adapter |
| `fusion: heuristic` | ③ + Heuristic Fusion | ④ + Adapter + Heuristic |
| `fusion: entropy` | ⑤ + Entropy Fusion | ⑥ + Adapter + Entropy ← **Final** |

---

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.3+ with CUDA 12.1 (cu121 build)
- NVIDIA GPU with 16 GB+ VRAM (80 GB A100 recommended for training)

### Environment

```bash
# Recommended: use the provided conda environment file
conda env create \
    --name segearth_stable \
    --file SegEarth-OV-3/env/environment.segearth_stable.yml
conda activate segearth_stable
```

Or manually:

```bash
conda create -n segearth python=3.10 -y
conda activate segearth
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm einops transformers pillow opencv-python scipy tqdm tensorboard
pip install -e SegEarth-OV-3/
```

### Weights

Place SAM3 weights at:
```
SegEarth-OV-3/weights/sam3/sam3.pt   # 3.3 GB
```

### Dataset (LoveDA)

Downloaded from HuggingFace. Urban and Rural scenes are mixed within each split.

```
datasets/LoveDA/
├── .cache/
├── urban:rural train images/   # RGB .png (1024×1024), urban + rural mixed
├── urban:rural train masks/    # Label .png (0–6), urban + rural mixed
├── urban:rural val images/     # RGB .png (1024×1024), urban + rural mixed
└── urban:rural val masks/      # Label .png (0–6), urban + rural mixed
```

---

## Training (PEFT)

### Basic run

```bash
cd SegEarth-OV-3/rs_adapter
python train_adapter.py \
    --segearthov3_path ~/capstone/SegEarth-OV-3 \
    --model_path       ~/capstone/SegEarth-OV-3/weights/sam3/sam3.pt \
    --data_root        ~/datasets/LoveDA \
    --dataset          loveda \
    --bottleneck       64 \
    --lr               1e-3 \
    --epochs           20 \
    --batch_size       4 \
    --seed             42 \
    --save_path        rs_adapter/adapter_ckpt.pt \
    --log_dir          rs_adapter/runs/adapter
```

### Ablation flags

```bash
# Baseline (no adapter, no FPN)
python train_adapter.py ... --no_adapter --no_fpn

# FPN only
python train_adapter.py ... --no_adapter

# Adapter only (no FPN)
python train_adapter.py ... --no_fpn

# Full (default): FPN + Adapter
python train_adapter.py ...
```

### Resume

```bash
python train_adapter.py ... --resume rs_adapter/adapter_ckpt.pt
```

### Monitor

```bash
tensorboard --logdir rs_adapter/runs --port 6006
```

### VRAM budget

| Config | Estimated VRAM |
|--------|---------------|
| batch=2, 1008² | ~18 GB |
| batch=4, 1008² | ~28–30 GB |

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
adapter_path: "/path/to/adapter_ckpt.pt"
adapter_bottleneck: 64
```

### Fusion modes

| `fusion_mode` | Description |
|---|---|
| `max` | Element-wise MAX (original SegEarth-OV-3) |
| `heuristic` | Fixed α per things/stuff category |
| `entropy` | Dynamic pixel-wise α from prediction entropy |

### Adapter modes

| `adapter_mode` | Description |
|---|---|
| `off` | Original SegEarth-OV-3 (no adapter, no FPN) |
| `full` | Load pre-trained RSAdapter + RSMultiscaleFPN |

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `sam3_model.backbone.forward_image()` called directly | `Sam3Processor.set_image()` has `@inference_mode()` decorator that blocks gradients |
| `hook_feats` stored **without** `.detach()` | Gradient must flow back through FPN → Adapter |
| `scale = zeros(1)` init in RSAdapter | Identity at start; avoids disrupting pretrained SAM3 features early in training |
| `adapted.to(x.dtype)` | Prevents float32/bfloat16 mismatch in mixed precision training |
| CrossEntropy + 0.5 × Soft Dice | CE handles class imbalance; Dice improves boundary IoU |
| `--no_fpn` path uses SAM3 native `vision_features` (72×72) | Cleanly isolates Adapter-only contribution without multi-scale |

---

## Roadmap

- [x] RSAdapter + RSMultiscaleFPN implementation
- [x] PEFT training pipeline (LoveDA, iSAID support)
- [x] Category-Adaptive Fusion (heuristic + entropy)
- [x] X-AnyLabeling integration
- [x] Closed-set ablation: complete 4-way comparison (Baseline 0.4905 / +FPN 0.6162 / +Adapter 0.6324 / +FPN+Adapter 0.6384)
- [ ] OVSS baseline evaluation on LoveDA
- [ ] OVSS + Adapter: measure domain adaptation effect on zero-shot setting
- [ ] Full ablation table: Fusion × Adapter × OVSS
- [ ] Concept Bank: hierarchical prompt + synonym expansion
- [ ] Concept Bank: local presence-guided filtering per sliding-window patch

---

## References

- SegEarth-OV-3: [earth-insights/SegEartg-OV-3](https://github.com/earth-insights/SegEarth-OV-3)
- SAM3: [facebookresearch/sam3] (https://github.com/facebookresearch/sam3)
- X-AnyLabeling: [CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- LoveDA Dataset: [LoveDA: A Remote Sensing Land-Cover Dataset](https://github.com/Junjue-Wang/LoveDA)
- iSAID Dataset: [iSAID: A Large-scale Dataset for Instance Segmentation](https://captain-whu.github.io/iSAID/)
