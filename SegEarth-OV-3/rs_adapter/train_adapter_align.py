"""
train_adapter_align.py — Phase 4: RSAdapter + Visual-Text Alignment Loss 학습

Phase 1(train_adapter.py)과의 차이점:
  - SAM3 language backbone(forward_text)을 직접 호출하여 클래스별 text feature 사전 추출
  - L_total = L_CE + 0.5 * L_Dice + α * L_align
  - L_align: Adapter 적용 후 visual feature와 SAM3 text feature 간 cosine similarity 최대화
  - SAM3 코드 수정 없음 — forward_text()는 @inference_mode() 데코레이터 없음
  - ablation 플래그(--no_adapter, --no_fpn) 제거 (align variant는 항상 Adapter 사용)

배경:
  Phase 3에서 CE+Dice만으로 학습한 Adapter를 OVSS에 얹으면 mIoU가 오히려 하락
  (47.38 → 46.70). 원인: proxy task gap — CE loss는 SAM3 cross-modal decoder가
  기대하는 visual-text alignment를 보장하지 않아 feature 분포가 왜곡됨.
  특히 water↑(+19.02) agricultural↓(-20.24) 트레이드오프가 대표적 증거.

실행 예시:
  cd ~/capstone/SegEarth-OV-3
  python rs_adapter/train_adapter_align.py \\
      --segearthov3_path /root/capstone-project/SegEarth-OV-3 \\
      --model_path      weights/sam3/sam3.pt \\
      --data_root       /root/capstone-project/datasets/LoveDA \\
      --dataset         loveda \\
      --align_weight    0.5 \\
      --epochs          20 \\
      --save_path       rs_adapter/ckpt_align_best.pt \\
      --log_dir         runs/adapter_align \\
      2>&1 | tee train_phase4_log.txt
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
parser.add_argument("--align_weight", type=float, default=0.5,
                    help="Alignment loss 가중치 α (L_total = L_CE + 0.5*L_Dice + α*L_align)")
parser.add_argument("--align_skip_bg", action="store_true", default=True,
                    help="Alignment loss에서 background 클래스 제외 (기본 True)")
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
parser.add_argument("--save_path", default="rs_adapter/ckpt_align.pt",
                    help="체크포인트 저장 경로 (best는 _best.pt로 자동 저장)")
parser.add_argument("--resume", default="",
                    help="이어 학습할 체크포인트 경로")
parser.add_argument("--log_dir", default="./runs/adapter_align",
                    help="TensorBoard 로그 디렉토리")
parser.add_argument("--seed", type=int, default=42,
                    help="재현성을 위한 글로벌 랜덤 시드")
args = parser.parse_args()

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


def alignment_loss(
    vision_features: torch.Tensor,
    masks_gt: torch.Tensor,
    text_feat_dict: dict,
    class_names: list,
    skip_bg: bool = True,
) -> torch.Tensor:
    """
    Adapter가 수정한 visual feature와 SAM3 text feature 간 cosine similarity를
    클래스별로 최대화하는 loss.

    Args:
        vision_features: (B, 256, H_f, W_f) — backbone vision_features (float)
                         H_f=W_f=72 (SAM3 기본 해상도에서의 feature map 크기)
        masks_gt:        (B, H, W) long — GT segmentation (0~C-1 범위, 학습 해상도)
        text_feat_dict:  {class_name: Tensor(256)} — 사전 추출된 text feature (float32)
        class_names:     list[str] — 데이터셋 클래스 이름 목록
        skip_bg:         background 클래스(index 0) 제외 여부

    Returns:
        scalar loss tensor (gradient 흐름 보장)
    """
    B, C_feat, H_f, W_f = vision_features.shape

    # GT를 vision_features 해상도(72×72)로 다운샘플
    gt_small = F.interpolate(
        masks_gt.float().unsqueeze(1),
        size=(H_f, W_f),
        mode="nearest",
    ).squeeze(1).long()  # (B, H_f, W_f)

    total_loss = vision_features.new_zeros(1).squeeze()
    count = 0

    start_cls = 1 if skip_bg else 0

    for b in range(B):
        vis = vision_features[b].float()  # (256, H_f, W_f), float32로 캐스트

        for cls_idx in range(start_cls, len(class_names)):
            pixel_mask = (gt_small[b] == cls_idx)  # (H_f, W_f)
            if pixel_mask.sum() == 0:
                continue

            # 해당 클래스 픽셀의 visual feature 평균: (256,)
            vis_cls = vis[:, pixel_mask].mean(dim=1)

            # 사전 추출된 text feature: (256,)
            text_cls = text_feat_dict[class_names[cls_idx]]

            sim = F.cosine_similarity(vis_cls.unsqueeze(0), text_cls.unsqueeze(0))
            total_loss = total_loss + (1.0 - sim.squeeze())
            count += 1

    if count == 0:
        return vision_features.new_zeros(1).squeeze()

    return total_loss / count


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
print("[1/5] SAM3 모델 로드 중...")
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
print("[2/5] 데이터셋 준비 중...")
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

# ── Text Feature 사전 추출 (학습 전 1회) ─────────────────────────────────────
# forward_text()는 vl_combiner.py에 정의되며 @inference_mode() 없음
# SAM3 language backbone은 frozen이므로 no_grad로 1회만 추출
print("[3/5] SAM3 text feature 사전 추출 중...")
text_feat_dict = {}
with torch.no_grad():
    for cls_name in class_names:
        text_out = sam3_model.backbone.forward_text([cls_name], device=str(device))
        # language_features: [seq_len, 1, 256] — 시퀀스 및 배치 차원 평균 → [256]
        feat = text_out["language_features"][:, 0, :].mean(dim=0).float().to(device)
        text_feat_dict[cls_name] = feat

skip_str = "(background 제외)" if args.align_skip_bg else "(background 포함)"
print(f"  {len(text_feat_dict)}개 클래스 text feature 추출 완료 {skip_str}")
print(f"  Alignment loss 가중치 α = {args.align_weight}")

# ── Adapter + FPN + 분류 헤드 초기화 ─────────────────────────────────────────
print("[4/5] Adapter + FPN + 분류 헤드 초기화 중...")
vit_blocks = sam3_model.backbone.vision_backbone.trunk.blocks  # len=32

adapters = nn.ModuleList([
    RSAdapter(d_model=1024, bottleneck=args.bottleneck)
    for _ in range(len(vit_blocks))
]).to(device)

fpn     = RSMultiscaleFPN(in_channels=1024, out_channels=256).to(device)
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
hooks = []
checkpoint_blocks = {7: "f7", 15: "f15", 23: "f23", 31: "f31"}

def make_hook(adapter_module: nn.Module, feat_key: str | None):
    def _hook(_module, _input, output):
        adapted = adapter_module(output)
        if feat_key is not None:
            hook_feats[feat_key] = adapted
        return adapted
    return _hook

for blk_idx in range(len(vit_blocks)):
    feat_key = checkpoint_blocks.get(blk_idx)
    hooks.append(
        vit_blocks[blk_idx].register_forward_hook(
            make_hook(adapters[blk_idx], feat_key)
        )
    )

# ── 옵티마이저 + 스케줄러 ─────────────────────────────────────────────────────
optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - start_epoch, eta_min=args.lr * 0.01
)

criterion_ce = nn.CrossEntropyLoss(ignore_index=255)

os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.log_dir)

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
print("[5/5] 학습 시작")
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, args.epochs):
    # ── Train ──────────────────────────────────────────────────────────────
    adapters.train()
    fpn.train()
    cls_head.train()

    epoch_loss       = 0.0
    epoch_loss_ce    = 0.0
    epoch_loss_dice  = 0.0
    epoch_loss_align = 0.0
    train_metric = MetricTracker(num_classes)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        hook_feats.clear()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            backbone_out = sam3_model.backbone.forward_image(images)

            fpn_out  = fpn(hook_feats)
            backbone_out["vision_features"] = fpn_out["p4"]
            backbone_out["backbone_fpn"] = [
                fpn_out["p2"], fpn_out["p3"], fpn_out["p4"], fpn_out["p5"],
            ]
            vis_feat = fpn_out["p4"].float()  # (B, 256, 72, 72)

            vis_up = F.interpolate(
                vis_feat,
                size=(args.resolution, args.resolution),
                mode="bilinear", align_corners=False,
            )
            logits = cls_head(vis_up)  # (B, C, 1008, 1008)

            loss_ce   = criterion_ce(logits, masks)
            loss_dice = dice_loss(logits, masks, num_classes)

        # Alignment loss — autocast 밖에서 float32로 계산
        # vision_features는 backbone의 p4 (Adapter hook 적용 후)
        # backbone_out["vision_features"]는 bfloat16이므로 .float() 변환은 alignment_loss 내부에서 처리
        loss_align = alignment_loss(
            vision_features=backbone_out["vision_features"],
            masks_gt=masks,
            text_feat_dict=text_feat_dict,
            class_names=class_names,
            skip_bg=args.align_skip_bg,
        )

        loss = loss_ce + 0.5 * loss_dice + args.align_weight * loss_align

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss       += loss.item()
        epoch_loss_ce    += loss_ce.item()
        epoch_loss_dice  += loss_dice.item()
        epoch_loss_align += loss_align.item()

        pred_np  = logits.detach().argmax(dim=1).cpu().numpy()
        label_np = masks.cpu().numpy()
        for b in range(pred_np.shape[0]):
            valid = label_np[b] != 255
            train_metric.update(pred_np[b][valid].ravel(), label_np[b][valid].ravel())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            align=f"{loss_align.item():.4f}",
        )

    scheduler.step()
    n_batches   = len(train_loader)
    avg_loss    = epoch_loss       / n_batches
    avg_ce      = epoch_loss_ce    / n_batches
    avg_dice    = epoch_loss_dice  / n_batches
    avg_align   = epoch_loss_align / n_batches
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
        f"Epoch [{epoch+1:>3}/{args.epochs}] "
        f"loss={avg_loss:.4f}  "
        f"(ce={avg_ce:.4f} dice={avg_dice:.4f} align={avg_align:.4f})  "
        f"train_mIoU={train_miou:.4f}  val_mIoU={val_miou:.4f}  "
        f"lr={scheduler.get_last_lr()[0]:.6f}"
    )
    writer.add_scalar("Loss/train_total", avg_loss,   epoch + 1)
    writer.add_scalar("Loss/ce",          avg_ce,     epoch + 1)
    writer.add_scalar("Loss/dice",        avg_dice,   epoch + 1)
    writer.add_scalar("Loss/align",       avg_align,  epoch + 1)
    writer.add_scalar("mIoU/train",       train_miou, epoch + 1)
    writer.add_scalar("mIoU/val",         val_miou,   epoch + 1)
    writer.add_scalar("LR",               scheduler.get_last_lr()[0], epoch + 1)

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
            "dataset":       args.dataset,
            "num_classes":   num_classes,
            "bottleneck":    args.bottleneck,
            "in_channels":   1024,
            "out_channels":  256,
            "resolution":    args.resolution,
            "align_weight":  args.align_weight,
            "align_skip_bg": args.align_skip_bg,
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
