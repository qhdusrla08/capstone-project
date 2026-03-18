"""
Fusion mode comparison test for SegEarth-OV-3 (Novelty Idea 1).

Compares fusion strategies on a single image and saves visualizations:
  - max          : element-wise max of instance agg & semantic head (original baseline)
  - inst_only    : heuristic α=1.0  → instance head only
  - sem_only     : heuristic α=0.0  → semantic head only
  - heuristic    : things α=0.8, stuff α=0.2
  - adaptive_split: things α=1.0 (inst only), stuff α=0.0 (sem only)
  - entropy      : per-pixel entropy-based adaptive weighting
                   (less confident head gets lower weight via binary entropy)

Usage:
    cd /home/yeon030108/capstone/SegEarth-OV-3
    python test_fusion_modes.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SEGEARTHOV3_DIR = "/home/yeon030108/capstone/SegEarth-OV-3"
IMAGE_PATH      = os.path.join(SEGEARTHOV3_DIR, "resources/oem_koeln_50.tif")
CLASSNAME_PATH  = os.path.join(SEGEARTHOV3_DIR, "configs/cls_openearthmap.txt")
MODEL_PATH      = os.path.join(SEGEARTHOV3_DIR, "weights/sam3/sam3.pt")
BPE_PATH        = os.path.join(SEGEARTHOV3_DIR, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
OUTPUT_DIR      = SEGEARTHOV3_DIR

# ── Inference hyperparams ─────────────────────────────────────────────────────
PROB_THD       = 0.0
BG_IDX         = 0
SLIDE_STRIDE   = 512
SLIDE_CROP     = 512
CONFIDENCE_THD = 0.5

# things 카테고리
THINGS_SET = {"building", "roof", "house"}

# ── cls_openearthmap.txt 기준 9-class colormap ────────────────────────────────
# background, bareland, grass, pavement, road, tree, water, cropland, building
CLASS_COLORS = np.array([
    [  0,   0,   0],  # 0 background   (black)
    [128,  96,  64],  # 1 bareland      (brown)
    [144, 210, 144],  # 2 grass         (light green)
    [192, 192, 192],  # 3 pavement      (gray)
    [255, 255, 255],  # 4 road          (white)
    [ 34, 100,  34],  # 5 tree          (dark green)
    [  0,  80, 200],  # 6 water         (blue)
    [200, 220, 100],  # 7 cropland      (yellow-green)
    [220,  60,  60],  # 8 building      (red)
], dtype=np.uint8)


# ── Helper ────────────────────────────────────────────────────────────────────
def get_cls_idx(path):
    with open(path) as f:
        name_sets = f.readlines()
    class_names, class_indices = [], []
    for idx, line in enumerate(name_sets):
        names = [n.strip().replace("\n", "") for n in line.split(",")]
        class_names += names
        class_indices += [idx] * len(names)
    return class_names, class_indices


def colorize(seg_pred, colors):
    rgb = np.zeros((*seg_pred.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colors):
        rgb[seg_pred == cls_idx] = color
    return rgb


# ── Inference (based on segearthov3_segmentor.py) ─────────────────────────────
def _inference_single_view(image, processor, query_words, num_queries, device,
                           fusion_mode, things_alpha, stuff_alpha):
    """단일 뷰(또는 crop 패치)에 대해 추론 수행."""
    w, h = image.size
    seg_logits = torch.zeros((num_queries, h, w), device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)

        for q_idx, query_word in enumerate(query_words):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=query_word)

            # ── Instance head: MAX_i(P_inst_i × score_i) ─────────────────
            if state["masks_logits"].shape[0] > 0:
                for inst_id in range(state["masks_logits"].shape[0]):
                    inst_logit = state["masks_logits"][inst_id].squeeze()
                    inst_score = state["object_score"][inst_id]
                    if inst_logit.shape != (h, w):
                        inst_logit = F.interpolate(
                            inst_logit.view(1, 1, *inst_logit.shape),
                            size=(h, w), mode="bilinear", align_corners=False,
                        ).squeeze()
                    seg_logits[q_idx] = torch.max(
                        seg_logits[q_idx], inst_logit * inst_score
                    )

            # ── Semantic head ─────────────────────────────────────────────
            sem_logit = state["semantic_mask_logits"]
            if sem_logit.shape != (h, w):
                sem_logit = F.interpolate(
                    sem_logit if sem_logit.dim() == 4
                    else sem_logit.unsqueeze(0).unsqueeze(0),
                    size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze()

            # ── Fusion ────────────────────────────────────────────────────
            inst_agg = seg_logits[q_idx].clone()

            if fusion_mode == "max":
                seg_logits[q_idx] = torch.max(inst_agg, sem_logit)
            elif fusion_mode == "entropy":
                # Per-pixel entropy-based adaptive weighting.
                # Binary entropy H(p) = -p·log(p) - (1-p)·log(1-p), range [0, 1].
                # Lower entropy → more confident → higher weight.
                # alpha_inst = (1 - H_inst) / (2 - H_inst - H_sem)
                EPS = 1e-6
                p_inst = torch.sigmoid(inst_agg.float())
                p_sem  = torch.sigmoid(sem_logit.float())

                def _binary_entropy(p):
                    p = p.clamp(EPS, 1.0 - EPS)
                    return -(p * p.log() + (1 - p) * (1 - p).log()) / np.log(2)  # [0,1]

                h_inst = _binary_entropy(p_inst)  # H×W, range [0,1]
                h_sem  = _binary_entropy(p_sem)

                conf_inst = 1.0 - h_inst          # confidence = 1 - entropy
                conf_sem  = 1.0 - h_sem
                total     = (conf_inst + conf_sem).clamp(min=EPS)
                alpha_map = conf_inst / total      # per-pixel weight for inst head

                seg_logits[q_idx] = (
                    alpha_map * inst_agg.float()
                    + (1.0 - alpha_map) * sem_logit.float()
                )
            else:
                # heuristic: things/stuff에 따라 서로 다른 α 적용
                cls_name = query_word.split(",")[0].strip().lower()
                alpha = things_alpha if cls_name in THINGS_SET else stuff_alpha
                seg_logits[q_idx] = (
                    alpha * inst_agg.float()
                    + (1.0 - alpha) * sem_logit.float()
                )

            # ── Presence score ─────────────────────────────────────────────
            seg_logits[q_idx] = seg_logits[q_idx] * state["presence_score"]

    return seg_logits


def _slide_inference(image, processor, query_words, num_queries, device,
                     fusion_mode, things_alpha, stuff_alpha):
    """슬라이딩 윈도우 추론."""
    w_img, h_img = image.size
    preds     = torch.zeros((num_queries, h_img, w_img), device=device)
    count_mat = torch.zeros((1, h_img, w_img), device=device)

    h_grids = max(h_img - SLIDE_CROP + SLIDE_STRIDE - 1, 0) // SLIDE_STRIDE + 1
    w_grids = max(w_img - SLIDE_CROP + SLIDE_STRIDE - 1, 0) // SLIDE_STRIDE + 1

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * SLIDE_STRIDE
            x1 = w_idx * SLIDE_STRIDE
            y2 = min(y1 + SLIDE_CROP, h_img)
            x2 = min(x1 + SLIDE_CROP, w_img)
            y1 = max(y2 - SLIDE_CROP, 0)
            x1 = max(x2 - SLIDE_CROP, 0)
            crop = image.crop((x1, y1, x2, y2))
            logit = _inference_single_view(
                crop, processor, query_words, num_queries, device,
                fusion_mode, things_alpha, stuff_alpha,
            )
            preds[:, y1:y2, x1:x2] += logit
            count_mat[:, y1:y2, x1:x2] += 1

    return preds / count_mat


def run_experiment(image, processor, query_words, query_idx_tensor,
                   num_cls, num_queries, device,
                   fusion_mode, things_alpha, stuff_alpha):
    """추론 실행 후 seg_pred (H×W numpy) 반환."""
    w, h = image.size
    if SLIDE_CROP > 0 and (SLIDE_CROP < w or SLIDE_CROP < h):
        seg_logits = _slide_inference(
            image, processor, query_words, num_queries, device,
            fusion_mode, things_alpha, stuff_alpha,
        )
    else:
        seg_logits = _inference_single_view(
            image, processor, query_words, num_queries, device,
            fusion_mode, things_alpha, stuff_alpha,
        )

    # synonym aggregation
    if num_cls != num_queries:
        seg_logits_u = seg_logits.unsqueeze(0)
        cls_index = F.one_hot(query_idx_tensor, num_classes=num_cls)
        cls_index = cls_index.T.view(num_cls, num_queries, 1, 1).float()
        seg_logits = (seg_logits_u * cls_index).max(1)[0]

    seg_pred = torch.argmax(seg_logits, dim=0)
    max_vals  = seg_logits.max(0)[0]
    seg_pred[max_vals < PROB_THD] = BG_IDX
    return seg_pred.cpu().numpy()


# ── Visualization ─────────────────────────────────────────────────────────────
def save_comparison(image, results, class_labels, filename="fusion_comparison.png"):
    """Input + 모든 실험 결과를 나란히 비교 저장."""
    n_cols = len(results) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 7))

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    for ax, (title, seg_pred) in zip(axes[1:], results):
        ax.imshow(colorize(seg_pred, CLASS_COLORS))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    patches = [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=class_labels[i])
        for i in range(len(class_labels))
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=len(class_labels),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    out = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


def save_individual(seg_pred, name):
    out = os.path.join(OUTPUT_DIR, f"fusion_{name}.png")
    plt.imsave(out, colorize(seg_pred, CLASS_COLORS))
    print(f"[saved] {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sys.path.insert(0, SEGEARTHOV3_DIR)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        checkpoint_path=MODEL_PATH,
        device=str(device),
    )
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THD, device=device)

    query_words, query_idx_list = get_cls_idx(CLASSNAME_PATH)
    num_cls     = max(query_idx_list) + 1
    num_queries = len(query_idx_list)
    query_idx_tensor = torch.tensor(query_idx_list, dtype=torch.int64, device=device)

    # 범례용 클래스명 (첫 번째 synonym만)
    with open(CLASSNAME_PATH) as f:
        class_labels = [line.split(",")[0].strip() for line in f.readlines()]

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image size: {image.size}")

    # ── 실험 정의 ─────────────────────────────────────────────────────────────
    # (제목, fusion_mode, things_alpha, stuff_alpha, 저장 파일명, comparison 포함 여부)
    experiments = [
        ("max\n(baseline)",          "max",       1.0, 1.0, "max",           True),
        ("inst only\n(α=1.0)",       "heuristic", 1.0, 1.0, "inst_only",     False),
        ("sem only\n(α=0.0)",        "heuristic", 0.0, 0.0, "sem_only",      False),
        ("heuristic\n(α=0.8)",       "heuristic", 0.8, 0.2, "Adaptive",      True),
        ("heuristic\n(α=0.7)",       "heuristic", 0.7, 0.3, "Adaptive_v2",      True),
        ("adaptive split\n(t=1,s=0)","heuristic", 1.0, 0.0, "adaptive_split",True),
        ("entropy\n(per-pixel)",     "entropy",   0.0, 0.0, "entropy",       True),
    ]

    results_all = []
    results_cmp = []
    for title, mode, t_alpha, s_alpha, fname, in_cmp in experiments:
        label = title.replace("\n", " ")
        print(f"\n[{label}] mode={mode}, things_alpha={t_alpha}, stuff_alpha={s_alpha}")
        seg_pred = run_experiment(
            image, processor, query_words, query_idx_tensor,
            num_cls, num_queries, device, mode, t_alpha, s_alpha,
        )
        save_individual(seg_pred, fname)
        results_all.append((title, seg_pred))
        if in_cmp:
            results_cmp.append((title, seg_pred))

    # 전체 결과 비교 (inst_only / sem_only 포함)
    save_comparison(image, results_all, class_labels, filename="fusion_comparison_v2.png")

    # 선별 비교: Input / max / heuristic(α=0.8) / adaptive(t=1,s=0) / entropy
    save_comparison(image, results_cmp, class_labels, filename="fusion_comparison_4_v2.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
