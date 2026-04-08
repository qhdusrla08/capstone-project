"""
train_adapter_distill_warmup.py — Phase 7: RSAdapter + Feature Distillation (β Warm-up)

train_adapter_distill.py(Phase 6)와의 차이점:
  - β를 고정값 대신 warm-up 스케줄로 동적 투입
  - 손실 공식: L_total = L_CE + 0.5*L_Dice + β_t * L_distill

β_t 스케줄 (LP-FT 전략, Kumar et al., ICLR 2022):
  ┌─────────────────────────────────────────────┐
  │  epoch < warmup_epochs         → β_t = 0    │
  │  warmup ≤ epoch < warmup+ramp  → β_t 선형↑  │
  │  epoch ≥ warmup + ramp         → β_t = β    │
  └─────────────────────────────────────────────┘

설계 근거:
  Phase 6 결과에서 background(-4.15), forest(-3.32) 하락 관찰.
  원인 가설: 학습 초반부터 distillation이 task loss(CE+Dice) 수렴을 방해,
  adapter가 LoveDA 도메인 표현을 충분히 학습하기 전에 SAM3 원본 특징 공간에 고정.
  → 초반 warmup_epochs 동안 task loss만으로 수렴 → 이후 distillation 점진 투입.

  참고 문헌:
    - LP-FT (Kumar et al., ICLR 2022): 사전학습 특징 보존을 위해
      linear probe → fine-tune 순서 적용 — 본 스크립트는 이를
      "task-first(β=0), distill-later(β 점진 증가)"로 변환.
    - CLIP-Adapter (Gao et al., 2021): residual adapter + feature regularization 조합.
    - ARC (Yadav et al., 2023): L2 regularization warm-up 전략.

실행 예시:
  cd ~/capstone/SegEarth-OV-3
  python rs_adapter/train_adapter_distill_warmup.py \\
      --segearthov3_path /root/capstone-project/SegEarth-OV-3 \\
      --model_path      weights/sam3/sam3.pt \\
      --data_root       /root/capstone-project/datasets/LoveDA \\
      --dataset         loveda \\
      --distill_weight  0.1 \\
      --warmup_epochs   5 \\
      --rampup_epochs   10 \\
      --distill_blocks  last \\
      --epochs          20 \\
      --save_path       rs_adapter/ckpt_distill_warmup.pt \\
      --log_dir         runs/adapter_distill_warmup \\
      2>&1 | tee train_phase7_log.txt
"""

import sys
import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm

# ── 인수 파싱 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--segearthov3_path", required=True,
                    help="SegEarth-OV-3 프로젝트 루트 경로")
parser.add_argument("--model_path", required=True,
                    help="SAM3 weights 경로 (sam3.pt)")
parser.add_argument("--data_root", required=True,
                    help="데이터셋 루트 경로")
parser.add_argument("--dataset", default="loveda",
                    choices=["loveda", "isaid"],
                    help="학습 데이터셋 선택")
parser.add_argument("--bottleneck", type=int, default=64,
                    help="RSAdapter 병목 차원")
parser.add_argument("--distill_weight", type=float, default=0.1,
                    help="Distillation loss 최종 가중치 β_max\n"
                         "(warm-up 완료 후 고정되는 값)")
parser.add_argument("--warmup_epochs", type=int, default=5,
                    help="β=0 유지 구간 (epoch 수). task loss만으로 수렴.\n"
                         "LP-FT의 linear probe 단계에 대응.")
parser.add_argument("--rampup_epochs", type=int, default=10,
                    help="β를 0 → β_max 까지 선형 증가하는 구간 (epoch 수).\n"
                         "warmup_epochs 직후부터 시작.")
parser.add_argument("--distill_blocks", type=str, default="last",
                    choices=["last", "checkpoints", "all"],
                    help="Distillation을 적용할 ViT 블록 범위:\n"
                         "  last        : block 31만 (SAM3 neck이 사용하는 블록, 권장)\n"
                         "  checkpoints : block 7, 15, 23, 31 (FPN checkpoint 블록)\n"
                         "  all         : block 0~31 전체 (가장 강한 제약)")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="학습률 (AdamW)")
parser.add_argument("--epochs", type=int, default=20,
                    help="총 에포크 수")
parser.add_argument("--batch_size", type=int, default=4,
                    help="배치 크기")
parser.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker 수")
parser.add_argument("--resolution", type=int, default=1008,
                    help="SAM3 입력 해상도 (기본 1008)")
parser.add_argument("--save_path", default="rs_adapter/ckpt_distill_warmup.pt",
                    help="체크포인트 저장 경로 (best는 _best.pt로 자동 저장)")
parser.add_argument("--resume", default="",
                    help="이어 학습할 체크포인트 경로")
parser.add_argument("--log_dir", default="./runs/adapter_distill_warmup",
                    help="TensorBoard 로그 디렉토리")
parser.add_argument("--seed", type=int, default=42,
                    help="재현성을 위한 글로벌 랜덤 시드")
parser.add_argument("--label_smoothing", type=float, default=0.1,
                    help="CE loss label smoothing factor (Müller et al., NeurIPS 2019)")
args = parser.parse_args()

# ── 인수 검증 ────────────────────────────────────────────────────────────────
if args.warmup_epochs + args.rampup_epochs > args.epochs:
    parser.error(
        f"warmup_epochs({args.warmup_epochs}) + rampup_epochs({args.rampup_epochs}) "
        f"> epochs({args.epochs}). β_max에 도달하기 전에 학습이 종료됩니다. "
        f"epochs를 늘리거나 warmup/rampup을 줄여주세요."
    )

# ── 시드 고정 ────────────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f"[Seed] {args.seed}")

# ── 경로 설정 ────────────────────────────────────────────────────────────────
sys.path.insert(0, args.segearthov3_path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam3 import build_sam3_image_model
from rs_adapter import RSAdapter, RSMultiscaleFPN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


# ── β 스케줄 계산 ─────────────────────────────────────────────────────────────
def compute_beta(epoch: int, warmup: int, rampup: int, beta_max: float) -> float:
    """
    LP-FT 전략에 따른 epoch별 β_t 계산.

    Args:
        epoch   : 현재 에폭 (0-indexed)
        warmup  : β=0 유지 구간 길이
        rampup  : 선형 증가 구간 길이
        beta_max: 최종 목표 β 값

    Returns:
        float: 이 에폭에서 사용할 β_t

    근거 (Kumar et al., ICLR 2022):
        pre-trained 특징 보존을 위해 task loss를 먼저 수렴시킨 뒤
        regularization을 점진 투입하면 OOD 일반화 성능 향상.
    """
    if epoch < warmup:
        return 0.0
    elif epoch < warmup + rampup:
        progress = (epoch - warmup) / rampup  # [0, 1)
        return beta_max * progress
    else:
        return beta_max


# ── Dataset 정의 ─────────────────────────────────────────────────────────────

class LoveDADataset(Dataset):
    NUM_CLASSES = 7
    CLASS_NAMES = ["Background", "Building", "Road", "Water",
                   "Barren", "Forest", "Agricultural"]

    def __init__(self, root: str, split: str = "Train", resolution: int = 1008,
                 augment: bool = False):
        self.resolution = resolution
        self.augment    = augment
        split_map = {"Train": "train", "Val": "validation"}
        split_dir = split_map.get(split, split.lower())
        self.img_paths, self.mask_paths = [], []
        for domain in ["Urban", "Rural"]:
            imgs  = sorted(glob.glob(os.path.join(root, split_dir, domain, "images_png", "*.png")))
            masks = sorted(glob.glob(os.path.join(root, split_dir, domain, "masks_png", "*.png")))
            self.img_paths.extend(imgs)
            self.mask_paths.extend(masks)
        assert len(self.img_paths) == len(self.mask_paths), \
            f"이미지({len(self.img_paths)})와 마스크({len(self.mask_paths)}) 수 불일치"

        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.augment:
            scale = random.choice([0.75, 1.0, 1.25, 1.5])
            orig_w, orig_h = img.size
            scaled_h = int(orig_h * scale)
            scaled_w = int(orig_w * scale)
            img  = img.resize((scaled_w, scaled_h), Image.BILINEAR)
            mask = mask.resize((scaled_w, scaled_h), Image.NEAREST)
            if scaled_h >= self.resolution and scaled_w >= self.resolution:
                top  = random.randint(0, scaled_h - self.resolution)
                left = random.randint(0, scaled_w - self.resolution)
                img  = img.crop((left, top, left + self.resolution, top + self.resolution))
                mask = mask.crop((left, top, left + self.resolution, top + self.resolution))
            else:
                img  = img.resize((self.resolution, self.resolution), Image.BILINEAR)
                mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
        else:
            img  = img.resize((self.resolution, self.resolution), Image.BILINEAR)
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)

        img_t   = self.normalize(v2.functional.to_image(img).to(torch.uint8))
        mask_np = np.array(mask, dtype=np.int64)
        # LoveDA 공식 포맷: 0=nodata, 1=background, ..., 7=agricultural
        # mmseg reduce_zero_label 동일 처리: 0→255(ignore), 1~7→0~6
        nodata  = (mask_np == 0)
        mask_np = mask_np - 1
        mask_np[nodata] = 255
        mask_t  = torch.from_numpy(mask_np)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img_t  = v2.functional.horizontal_flip(img_t)
                mask_t = v2.functional.horizontal_flip(mask_t.unsqueeze(0)).squeeze(0)
            if torch.rand(1).item() < 0.5:
                img_t  = v2.functional.vertical_flip(img_t)
                mask_t = v2.functional.vertical_flip(mask_t.unsqueeze(0)).squeeze(0)
            k = random.randint(0, 3)                                # 0°/90°/180°/270°
            if k > 0:
                img_t  = torch.rot90(img_t,  k, dims=[1, 2])       # (C, H, W)
                mask_t = torch.rot90(mask_t, k, dims=[0, 1])       # (H, W)

        return img_t, mask_t


class ISAIDDataset(Dataset):
    NUM_CLASSES = 15
    CLASS_NAMES = ["Background", "Ship", "Storage_tank", "Baseball_diamond",
                   "Tennis_court", "Basketball_court", "Ground_track_field",
                   "Bridge", "Large_vehicle", "Small_vehicle", "Helicopter",
                   "Swimming_pool", "Roundabout", "Soccer_field", "Plane"]

    def __init__(self, root: str, split: str = "train", resolution: int = 1008):
        self.resolution = resolution
        img_dir  = os.path.join(root, split, "images")
        mask_dir = os.path.join(root, split, "semantic_png")
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), \
            f"이미지/마스크 수 불일치: {len(self.img_paths)} vs {len(self.mask_paths)}"

        self.img_transform = v2.Compose([
            v2.Resize((resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        img_t = self.img_transform(
            v2.functional.to_image(img).to(torch.uint8)
        )
        mask_np = np.array(
            mask.resize((self.resolution, self.resolution), Image.NEAREST),
            dtype=np.int64
        )
        mask_np = np.clip(mask_np, 0, self.NUM_CLASSES - 1)
        return img_t, torch.from_numpy(mask_np)


def build_dataset(args):
    if args.dataset == "loveda":
        train_ds = LoveDADataset(args.data_root, split="Train", resolution=args.resolution, augment=True)
        val_ds   = LoveDADataset(args.data_root, split="Val",   resolution=args.resolution, augment=False)
        return train_ds, val_ds, LoveDADataset.NUM_CLASSES, LoveDADataset.CLASS_NAMES
    elif args.dataset == "isaid":
        train_ds = ISAIDDataset(args.data_root, split="train", resolution=args.resolution)
        val_ds   = ISAIDDataset(args.data_root, split="val",   resolution=args.resolution)
        return train_ds, val_ds, ISAIDDataset.NUM_CLASSES, ISAIDDataset.CLASS_NAMES
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# ── 손실 함수 ─────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, tgt: torch.Tensor,
              num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    valid     = tgt != ignore_index
    tgt_clean = tgt.clone()
    tgt_clean[~valid] = 0
    pred_soft = pred.softmax(dim=1)
    tgt_oh    = F.one_hot(tgt_clean, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1).float()
    pred_soft  = pred_soft * valid_mask
    tgt_oh     = tgt_oh    * valid_mask
    intersection = (pred_soft * tgt_oh).sum(dim=(0, 2, 3))
    cardinality  = (pred_soft + tgt_oh).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1e-6) / (cardinality + 1e-6)).mean()


def distillation_loss(pairs: list) -> torch.Tensor:
    """
    Adapter 적용 후 ViT 블록 출력과 원본 출력 간 cosine dissimilarity를 최소화.

    OVSS 추론 경로 (eval_with_adapter.py):
        ViT block 출력 (adapter hook 수정됨) → SAM3 original neck → 256-dim → cross-modal decoder
    SAM3 neck은 마지막 ViT 블록(block 31) 출력만 사용 (necks.py: x = xs[-1]).
    따라서 block 31의 distillation이 OVSS 경로에 직접 작용한다.

    β_t=0인 warm-up 구간에서도 이 함수는 호출되지만, 호출 측에서 β_t를 곱해
    loss에 반영하므로 실제로는 gradient가 흐르지 않는다.

    Args:
        pairs: list of (adapted, original)
            adapted:  Adapter 적용 후 ViT 블록 출력 — (B, H, W, C) 또는 (B, N, C)
            original: Adapter 적용 전 원본 출력 (detached) — 동일 shape

    Returns:
        scalar loss tensor
    """
    if not pairs:
        return torch.tensor(0.0, device=device)

    total = pairs[0][0].new_zeros(1).squeeze()
    for adapted, original in pairs:
        B = adapted.shape[0]
        C = adapted.shape[-1]
        a = adapted.reshape(B, -1, C).float()
        o = original.reshape(B, -1, C).float()
        sim   = F.cosine_similarity(a, o, dim=-1)  # (B, N)
        total = total + (1.0 - sim.mean())

    return total / len(pairs)


# ── mIoU 계산 ─────────────────────────────────────────────────────────────────

class MetricTracker:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray):
        mask = (label >= 0) & (label < self.num_classes)
        idx  = self.num_classes * label[mask] + pred[mask]
        self.conf_mat += np.bincount(idx, minlength=self.num_classes ** 2)\
                           .reshape(self.num_classes, self.num_classes)

    def miou(self) -> float:
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        iou   = tp / np.maximum(tp + fp + fn, 1e-6)
        valid = (self.conf_mat.sum(axis=1) > 0)
        return float(iou[valid].mean()) if valid.any() else 0.0

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        return np.where(
            self.conf_mat.sum(axis=1) > 0,
            tp / np.maximum(tp + fp + fn, 1e-6),
            np.nan,
        )

    def reset(self):
        self.conf_mat[:] = 0


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
print("[1/4] SAM3 모델 로드 중...")
bpe_path = os.path.join(args.segearthov3_path, "sam3", "assets",
                        "bpe_simple_vocab_16e6.txt.gz")
sam3_model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=args.model_path,
    device=str(device),
)
sam3_model.to(device)

for p in sam3_model.parameters():
    p.requires_grad = False
sam3_model.eval()

# ── DataLoader 구성 ───────────────────────────────────────────────────────────
print("[2/4] 데이터셋 준비 중...")
train_ds, val_ds, num_classes, class_names = build_dataset(args)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, drop_last=True,
    worker_init_fn=seed_worker, generator=g,
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    worker_init_fn=seed_worker,
)
print(f"  Train: {len(train_ds)}장 | Val: {len(val_ds)}장 | Classes: {num_classes}")

# ── Distillation target 블록 결정 ─────────────────────────────────────────────
if args.distill_blocks == "last":
    target_block_set = {31}
elif args.distill_blocks == "checkpoints":
    target_block_set = {7, 15, 23, 31}
elif args.distill_blocks == "all":
    target_block_set = set(range(32))
else:
    raise ValueError(f"Unknown distill_blocks: {args.distill_blocks}")

print(f"  Distillation target blocks: {sorted(target_block_set)}")
print(f"  β_max = {args.distill_weight}  |  "
      f"warmup = {args.warmup_epochs} epoch  |  "
      f"rampup = {args.rampup_epochs} epoch")
print(f"  β=0 구간: epoch 1~{args.warmup_epochs}  |  "
      f"β 선형 증가: epoch {args.warmup_epochs+1}~{args.warmup_epochs+args.rampup_epochs}  |  "
      f"β={args.distill_weight} 고정: epoch {args.warmup_epochs+args.rampup_epochs+1}~{args.epochs}")

# ── Adapter + FPN + 분류 헤드 초기화 ─────────────────────────────────────────
print("[3/4] Adapter + FPN + 분류 헤드 초기화 중...")
vit_blocks = sam3_model.backbone.vision_backbone.trunk.blocks  # len=32

adapters = nn.ModuleList([
    RSAdapter(d_model=1024, bottleneck=args.bottleneck)
    for _ in range(len(vit_blocks))
]).to(device)

fpn      = RSMultiscaleFPN(in_channels=1024, out_channels=256).to(device)
cls_head = nn.Conv2d(256, num_classes, kernel_size=1).to(device)

trainable_params = (
    list(adapters.parameters())
    + list(fpn.parameters())
    + list(cls_head.parameters())
)
total_trainable = sum(p.numel() for p in trainable_params)
print(f"  학습 파라미터: {total_trainable / 1e6:.2f}M / SAM3 동결")

# ── 이어 학습 ────────────────────────────────────────────────────────────────
start_epoch = 0
best_miou   = 0.0
if args.resume and os.path.isfile(args.resume):
    ckpt = torch.load(args.resume, map_location=device)
    adapters.load_state_dict(ckpt["adapters"])
    fpn.load_state_dict(ckpt["fpn"])
    cls_head.load_state_dict(ckpt["cls_head"])
    start_epoch = ckpt.get("epoch", 0)
    best_miou   = ckpt.get("best_miou", 0.0)
    print(f"  체크포인트 로드: epoch={start_epoch}, best_mIoU={best_miou:.4f}")

# ── Forward Hook 등록 ────────────────────────────────────────────────────────
hook_feats: dict = {}
distill_pairs: list = []
hooks = []
checkpoint_blocks = {7: "f7", 15: "f15", 23: "f23", 31: "f31"}


def make_hook(adapter_module: nn.Module,
              feat_key: str | None,
              is_distill_target: bool):
    """
    ViT 블록의 forward hook 생성.

    - adapter_module    : 해당 블록에 대응하는 RSAdapter
    - feat_key          : hook_feats에 저장할 키 (FPN 입력용, None이면 저장 안 함)
    - is_distill_target : True이면 (adapted, original) 쌍을 distill_pairs에 추가

    hook의 반환값이 ViT 블록의 실제 출력을 대체하므로 이후 블록들은
    adapter가 수정한 feature를 입력으로 받는다.
    SAM3 original neck(necks.py)도 hook 적용 후의 마지막 블록 출력을 사용한다.

    warm-up 구간(β_t=0)에서도 distill_pairs는 채워지지만, 학습 루프에서
    β_t=0 을 곱하므로 실제로는 gradient가 adapter로 흐르지 않는다.
    """
    def _hook(_module, _input, output):
        adapted = adapter_module(output)
        if is_distill_target:
            distill_pairs.append((adapted, output.detach()))
        if feat_key is not None:
            hook_feats[feat_key] = adapted
        return adapted
    return _hook


for blk_idx in range(len(vit_blocks)):
    feat_key       = checkpoint_blocks.get(blk_idx)
    is_distill_tgt = blk_idx in target_block_set
    hooks.append(
        vit_blocks[blk_idx].register_forward_hook(
            make_hook(adapters[blk_idx], feat_key, is_distill_tgt)
        )
    )

# ── 옵티마이저 + 스케줄러 ─────────────────────────────────────────────────────
optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - start_epoch, eta_min=args.lr * 0.01
)

# ── 클래스 가중치 (median frequency balancing) ────────────────────────────────
# LoveDA: Wang et al. (NeurIPS 2021) 기반 근사 픽셀 빈도 → median / freq_c
# iSAID: 통계 미확보 → uniform
def get_class_weights(dataset_name: str, num_classes: int) -> torch.Tensor:
    if dataset_name == "loveda":
        freq = torch.tensor([0.27, 0.14, 0.11, 0.14, 0.03, 0.14, 0.17])
        return freq.median() / freq
    return torch.ones(num_classes)

class_weights = get_class_weights(args.dataset, num_classes).to(device)
criterion_ce = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=255,
    label_smoothing=args.label_smoothing,
)

os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.log_dir)

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
print("[4/4] 학습 시작")
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, args.epochs):
    # β_t 계산 — 에폭 시작 시 결정
    beta_t = compute_beta(epoch, args.warmup_epochs, args.rampup_epochs,
                          args.distill_weight)
    phase_label = (
        "warmup"  if epoch < args.warmup_epochs else
        "rampup"  if epoch < args.warmup_epochs + args.rampup_epochs else
        "stable"
    )

    # ── Train ──────────────────────────────────────────────────────────────
    adapters.train()
    fpn.train()
    cls_head.train()

    epoch_loss         = 0.0
    epoch_loss_ce      = 0.0
    epoch_loss_dice    = 0.0
    epoch_loss_distill = 0.0
    train_metric = MetricTracker(num_classes)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{args.epochs} [Train|{phase_label}|β={beta_t:.4f}]",
        leave=False,
    )

    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        hook_feats.clear()
        distill_pairs.clear()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            backbone_out = sam3_model.backbone.forward_image(images)

            fpn_out  = fpn(hook_feats)
            backbone_out["vision_features"] = fpn_out["p4"]  # dead code
            backbone_out["backbone_fpn"] = [
                fpn_out["p2"], fpn_out["p3"], fpn_out["p4"], fpn_out["p5"],
            ]

            vis_feat = fpn_out["p4"].float()  # (B, 256, 288, 288)
            vis_up   = F.interpolate(
                vis_feat,
                size=(args.resolution, args.resolution),
                mode="bilinear", align_corners=False,
            )
            logits = cls_head(vis_up)  # (B, C, 1008, 1008)

            loss_ce   = criterion_ce(logits, masks)
            loss_dice = dice_loss(logits, masks, num_classes)

        # Distillation loss — autocast 밖에서 float32로 계산
        loss_distill = distillation_loss(distill_pairs)

        # β_t 적용: warm-up 구간(β_t=0)에서는 distill gradient 차단
        loss = loss_ce + 0.5 * loss_dice + beta_t * loss_distill

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss         += loss.item()
        epoch_loss_ce      += loss_ce.item()
        epoch_loss_dice    += loss_dice.item()
        epoch_loss_distill += loss_distill.item()

        pred_np  = logits.detach().argmax(dim=1).cpu().numpy()
        label_np = masks.cpu().numpy()
        for b in range(pred_np.shape[0]):
            valid = label_np[b] != 255
            train_metric.update(pred_np[b][valid].ravel(), label_np[b][valid].ravel())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            distill=f"{loss_distill.item():.4f}",
            beta=f"{beta_t:.4f}",
        )

    scheduler.step()
    n_batches   = len(train_loader)
    avg_loss    = epoch_loss         / n_batches
    avg_ce      = epoch_loss_ce      / n_batches
    avg_dice    = epoch_loss_dice    / n_batches
    avg_distill = epoch_loss_distill / n_batches
    train_miou  = train_metric.miou()

    # ── Validation ───────────────────────────────────────────────────────
    adapters.eval()
    fpn.eval()
    cls_head.eval()
    val_metric = MetricTracker(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(val_loader,
                                  desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                                  leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            hook_feats.clear()
            distill_pairs.clear()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                backbone_out = sam3_model.backbone.forward_image(images)
                fpn_out  = fpn(hook_feats)
                vis_feat = fpn_out["p4"].float()
                vis_up   = F.interpolate(
                    vis_feat,
                    size=(args.resolution, args.resolution),
                    mode="bilinear", align_corners=False,
                )
                logits = cls_head(vis_up)

            pred_np  = logits.argmax(dim=1).cpu().numpy()
            label_np = masks.cpu().numpy()
            for b in range(pred_np.shape[0]):
                valid = label_np[b] != 255
                val_metric.update(pred_np[b][valid].ravel(), label_np[b][valid].ravel())

    val_miou = val_metric.miou()

    # ── 로깅 ────────────────────────────────────────────────────────────
    print(
        f"Epoch [{epoch+1:>3}/{args.epochs}] [{phase_label}] β={beta_t:.4f}  "
        f"loss={avg_loss:.4f}  "
        f"(ce={avg_ce:.4f} dice={avg_dice:.4f} distill={avg_distill:.4f})  "
        f"train_mIoU={train_miou:.4f}  val_mIoU={val_miou:.4f}  "
        f"lr={scheduler.get_last_lr()[0]:.6f}"
    )
    writer.add_scalar("Loss/train_total",  avg_loss,    epoch + 1)
    writer.add_scalar("Loss/ce",           avg_ce,      epoch + 1)
    writer.add_scalar("Loss/dice",         avg_dice,    epoch + 1)
    writer.add_scalar("Loss/distill",      avg_distill, epoch + 1)
    writer.add_scalar("mIoU/train",        train_miou,  epoch + 1)
    writer.add_scalar("mIoU/val",          val_miou,    epoch + 1)
    writer.add_scalar("LR",                scheduler.get_last_lr()[0], epoch + 1)
    writer.add_scalar("Beta/distill",      beta_t,      epoch + 1)  # β_t 추이 관찰

    per_cls_iou = val_metric.per_class_iou()
    print(f"  {'Class':<22} {'IoU':>6}")
    print(f"  {'-'*30}")
    for cls_idx, (name, iou_val) in enumerate(zip(class_names, per_cls_iou)):
        if np.isnan(iou_val):
            print(f"  {name:<22} {'N/A':>6}")
        else:
            print(f"  {name:<22} {iou_val:.4f}")
            writer.add_scalar(f"IoU_val/{name}", iou_val, epoch + 1)

    # ── 체크포인트 저장 ──────────────────────────────────────────────────
    ckpt = {
        "epoch":      epoch + 1,
        "adapters":   adapters.state_dict(),
        "fpn":        fpn.state_dict(),
        "cls_head":   cls_head.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "best_miou":  best_miou,
        "config": {
            "dataset":        args.dataset,
            "num_classes":    num_classes,
            "bottleneck":     args.bottleneck,
            "in_channels":    1024,
            "out_channels":   256,
            "resolution":     args.resolution,
            "distill_weight": args.distill_weight,
            "distill_blocks": args.distill_blocks,
            "warmup_epochs":  args.warmup_epochs,
            "rampup_epochs":  args.rampup_epochs,
        },
    }

    torch.save(ckpt, args.save_path)

    if val_miou > best_miou:
        best_miou = val_miou
        best_path = args.save_path.replace(".pt", "_best.pt")
        torch.save(ckpt, best_path)
        print(f"  ★ Best mIoU 갱신: {best_miou:.4f} → 저장: {best_path}")

# ── 정리 ──────────────────────────────────────────────────────────────────────
writer.close()
for h in hooks:
    h.remove()
print(f"\n학습 완료. 최종 체크포인트: {args.save_path}")
print(f"Best val mIoU: {best_miou:.4f}")
