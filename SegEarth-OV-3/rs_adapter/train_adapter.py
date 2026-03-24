"""
train_adapter.py — RSAdapter + RSMultiscaleFPN PEFT 학습 스크립트

학습 전략:
  - SAM3 파라미터 완전 동결 (requires_grad=False)
  - RSAdapter (32개, ~4.2M) + RSMultiscaleFPN (~3.4M) + ClsHead (~0.002M) 만 학습
  - 데이터셋: LoveDA (7클래스) 또는 iSAID (15클래스), --dataset 플래그로 선택
  - 손실: CrossEntropy + 0.5 × Dice Loss
  - Gradient flow: sam3_model.backbone.forward_image() 직접 호출
    (Sam3Processor.set_image()는 @inference_mode() 데코레이터로 gradient 차단되므로 우회)

실행 예시:
  python train_adapter.py \
      --segearthov3_path /path/to/SegEarth-OV-3 \
      --model_path      /path/to/sam3.pt \
      --data_root       /path/to/LoveDA \
      --dataset         loveda \
      --bottleneck      64 \
      --lr              1e-3 \
      --epochs          20 \
      --batch_size      2 \
      --save_path       ./adapter_ckpt.pt \
      --log_dir         ./runs/adapter
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

# ── 인수 파싱 ───────────────────────────────────────────────────────────────
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
parser.add_argument("--no_adapter", action="store_true",
                    help="Adapter 없이 FPN+Head만 학습 (baseline용)")
parser.add_argument("--no_fpn", action="store_true",
                    help="FPN 없이 SAM3 vision_features 직접 사용 (ablation용)")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="학습률 (AdamW)")
parser.add_argument("--epochs", type=int, default=20,
                    help="총 에포크 수")
parser.add_argument("--batch_size", type=int, default=2,
                    help="배치 크기 (A100 80GB 기준 4까지 안전)")
parser.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker 수")
parser.add_argument("--resolution", type=int, default=1008,
                    help="SAM3 입력 해상도 (기본 1008)")
parser.add_argument("--save_path", default="adapter_ckpt.pt",
                    help="체크포인트 저장 경로")
parser.add_argument("--resume", default="",
                    help="이어 학습할 체크포인트 경로")
parser.add_argument("--log_dir", default="./runs/adapter",
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
    """
    LoveDA: Urban + Rural, 7 클래스
      0=Background, 1=Building, 2=Road, 3=Water,
      4=Barren, 5=Forest, 6=Agricultural
    디렉토리 구조:
      {root}/{split}/Urban/images_png/*.png
      {root}/{split}/Urban/masks_png/*.png
      {root}/{split}/Rural/images_png/*.png
      {root}/{split}/Rural/masks_png/*.png
    """
    NUM_CLASSES = 7
    CLASS_NAMES = ["Background", "Building", "Road", "Water",
                   "Barren", "Forest", "Agricultural"]

    def __init__(self, root: str, split: str = "Train", resolution: int = 1008):
        self.resolution = resolution
        split_lower = split.lower()  # "Train" → "train", "Val" → "val"
        img_dir  = os.path.join(root, f"urban:rural {split_lower} images")
        mask_dir = os.path.join(root, f"urban:rural {split_lower} masks")
        self.img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), \
            f"이미지({len(self.img_paths)})와 마스크({len(self.mask_paths)}) 수 불일치"

        # SAM3와 동일한 전처리 (sam3_image_processor.py 참조)
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
        )                                                           # (3, H, W) float32
        mask_np = np.array(
            mask.resize((self.resolution, self.resolution), Image.NEAREST),
            dtype=np.int64
        )
        mask_np = np.clip(mask_np, 0, self.NUM_CLASSES - 1)        # 범위 보정
        mask_t = torch.from_numpy(mask_np)                          # (H, W) int64
        return img_t, mask_t


class ISAIDDataset(Dataset):
    """
    iSAID: 15 클래스 인스턴스+시맨틱 세그멘테이션
    디렉토리 구조:
      {root}/train/images/*.png
      {root}/train/semantic_png/*.png
    """
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
        train_ds = LoveDADataset(args.data_root, split="Train",
                                 resolution=args.resolution)
        val_ds   = LoveDADataset(args.data_root, split="Val",
                                 resolution=args.resolution)
        num_classes  = LoveDADataset.NUM_CLASSES
        class_names  = LoveDADataset.CLASS_NAMES
    elif args.dataset == "isaid":
        train_ds = ISAIDDataset(args.data_root, split="train",
                                resolution=args.resolution)
        val_ds   = ISAIDDataset(args.data_root, split="val",
                                resolution=args.resolution)
        num_classes  = ISAIDDataset.NUM_CLASSES
        class_names  = ISAIDDataset.CLASS_NAMES
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_ds, val_ds, num_classes, class_names


# ── 손실 함수 ────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, tgt: torch.Tensor,
              num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Soft Dice Loss (multi-class).
    pred: (B, C, H, W) float logits
    tgt:  (B, H, W)    long labels
    """
    valid = tgt != ignore_index
    tgt_clean = tgt.clone()
    tgt_clean[~valid] = 0

    pred_soft = pred.softmax(dim=1)                                 # (B, C, H, W)
    tgt_oh = F.one_hot(tgt_clean, num_classes=num_classes)          # (B, H, W, C)
    tgt_oh = tgt_oh.permute(0, 3, 1, 2).float()                    # (B, C, H, W)

    # ignore_index 위치 마스킹
    valid_mask = valid.unsqueeze(1).float()
    pred_soft  = pred_soft * valid_mask
    tgt_oh     = tgt_oh    * valid_mask

    intersection = (pred_soft * tgt_oh).sum(dim=(0, 2, 3))
    cardinality  = (pred_soft + tgt_oh).sum(dim=(0, 2, 3))
    dice_per_cls = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return 1.0 - dice_per_cls.mean()


# ── mIoU 계산 ────────────────────────────────────────────────────────────────

class MetricTracker:
    """배치 단위 confusion matrix를 누적해 epoch mIoU를 계산한다."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray):
        """pred, label: (H*W,) flattened, 범위 [0, num_classes)"""
        mask = (label >= 0) & (label < self.num_classes)
        idx  = self.num_classes * label[mask] + pred[mask]
        self.conf_mat += np.bincount(idx, minlength=self.num_classes ** 2)\
                           .reshape(self.num_classes, self.num_classes)

    def miou(self) -> float:
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1e-6)
        valid = (self.conf_mat.sum(axis=1) > 0)
        return float(iou[valid].mean()) if valid.any() else 0.0

    def per_class_iou(self) -> np.ndarray:
        """클래스별 IoU 배열 반환. 해당 클래스 GT가 없으면 NaN."""
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        iou = np.where(
            self.conf_mat.sum(axis=1) > 0,
            tp / np.maximum(tp + fp + fn, 1e-6),
            np.nan,
        )
        return iou

    def reset(self):
        self.conf_mat[:] = 0


# ── 모델 로드 ────────────────────────────────────────────────────────────────
print("[1/4] SAM3 모델 로드 중...")
bpe_path = os.path.join(args.segearthov3_path, "sam3", "assets",
                        "bpe_simple_vocab_16e6.txt.gz")
sam3_model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=args.model_path,
    device=str(device),
)
sam3_model.to(device)

# SAM3 파라미터 완전 동결
for p in sam3_model.parameters():
    p.requires_grad = False
sam3_model.eval()   # BN/Dropout 동결 (ViT에는 없지만 안전하게)

# ── DataLoader 구성 ──────────────────────────────────────────────────────────
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

# ── 어댑터 + FPN + 분류 헤드 초기화 ─────────────────────────────────────────
print("[3/4] 어댑터 + FPN + 분류 헤드 초기화 중...")
# ViT 블록 접근: backbone → vision_backbone(Sam3DualViTDetNeck) → trunk(ViT) → blocks
vit_blocks = sam3_model.backbone.vision_backbone.trunk.blocks  # nn.ModuleList, len=32

adapters = nn.ModuleList([
    RSAdapter(d_model=1024, bottleneck=args.bottleneck)
    for _ in range(len(vit_blocks))
]).to(device)

fpn = RSMultiscaleFPN(in_channels=1024, out_channels=256).to(device)

# 업샘플 + 1×1 Conv 분류 헤드 (vision_features: (B,256,72,72) → (B,C,1008,1008))
cls_head = nn.Conv2d(256, num_classes, kernel_size=1).to(device)

# 학습 가능 파라미터만 옵티마이저에 등록
trainable_params = (
    ([] if args.no_adapter else list(adapters.parameters()))
    + ([] if args.no_fpn else list(fpn.parameters()))
    + list(cls_head.parameters())
)
total_trainable = sum(p.numel() for p in trainable_params)
adapter_status = "비활성화" if args.no_adapter else "활성화"
fpn_status     = "비활성화" if args.no_fpn     else "활성화"
print(f"  학습 파라미터: {total_trainable / 1e6:.2f}M / SAM3 동결 / Adapter {adapter_status} / FPN {fpn_status}")

# ── 이어 학습 (resume) ───────────────────────────────────────────────────────
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
# gradient 흐름 보장: adapted 텐서를 .detach() 없이 저장
hook_feats: dict = {}
hooks = []
checkpoint_blocks = {7: "f7", 15: "f15", 23: "f23", 31: "f31"}

def make_hook(adapter_module: nn.Module, feat_key: str | None,
              use_adapter: bool = True, store_feat: bool = True):
    def _hook(_module, _input, output):
        adapted = adapter_module(output) if use_adapter else output  # Adapter bypass 가능
        if feat_key is not None and store_feat:
            hook_feats[feat_key] = adapted      # .detach() 없음 → FPN까지 gradient 흐름
        return adapted                          # 블록 출력 교체
    return _hook

for blk_idx in range(len(vit_blocks)):
    feat_key = checkpoint_blocks.get(blk_idx)
    hook = vit_blocks[blk_idx].register_forward_hook(
        make_hook(adapters[blk_idx], feat_key,
                  use_adapter=not args.no_adapter,
                  store_feat=not args.no_fpn)
    )
    hooks.append(hook)

# ── 옵티마이저 + 스케줄러 ─────────────────────────────────────────────────────
optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - start_epoch, eta_min=args.lr * 0.01
)

criterion_ce = nn.CrossEntropyLoss(ignore_index=255)

# ── TensorBoard ──────────────────────────────────────────────────────────────
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.log_dir)

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
print("[4/4] 학습 시작")
scaler = torch.cuda.amp.GradScaler()    # bfloat16 혼합정밀도용

for epoch in range(start_epoch, args.epochs):
    # ── Train ─────────────────────────────────────────────────────────────
    adapters.train()
    fpn.train()
    cls_head.train()

    epoch_loss   = 0.0
    train_metric = MetricTracker(num_classes)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                leave=False)

    for images, masks in pbar:
        images = images.to(device)      # (B, 3, 1008, 1008)
        masks  = masks.to(device)       # (B, 1008, 1008)

        optimizer.zero_grad()
        hook_feats.clear()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # SAM3 backbone 직접 호출 — @inference_mode() 우회 → gradient 흐름 보장
            # hook이 ViT 각 블록 forward 중 실행되어 hook_feats에 F7/F15/F23/F31 저장
            backbone_out = sam3_model.backbone.forward_image(images)

            if args.no_fpn:
                # FPN 없이 SAM3 native vision_features 직접 사용
                vis_feat = backbone_out["vision_features"].float()  # (B, 256, H, W)
            else:
                # FPN 퓨전: (B,1024,72,72) 중간 피처 → (B,256,72,72) P4
                fpn_out = fpn(hook_feats)
                backbone_out["vision_features"] = fpn_out["p4"]
                backbone_out["backbone_fpn"] = [
                    fpn_out["p2"],  # (B, 256, 288, 288)
                    fpn_out["p3"],  # (B, 256, 144, 144)
                    fpn_out["p4"],  # (B, 256, 72,  72)
                    fpn_out["p5"],  # (B, 256, 36,  36)
                ]
                vis_feat = fpn_out["p4"].float()

            # 분류 헤드: 업샘플 → 픽셀 분류
            vis_up = F.interpolate(
                vis_feat,
                size=(args.resolution, args.resolution),
                mode="bilinear", align_corners=False,
            )                                                       # (B, 256, 1008, 1008)
            logits = cls_head(vis_up)                               # (B, C, 1008, 1008)

            loss_ce   = criterion_ce(logits, masks)
            loss_dice = dice_loss(logits, masks, num_classes)
            loss      = loss_ce + 0.5 * loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 메트릭 누적
        epoch_loss += loss.item()
        pred_np  = logits.detach().argmax(dim=1).cpu().numpy()
        label_np = masks.cpu().numpy()
        for b in range(pred_np.shape[0]):
            valid = label_np[b] != 255
            train_metric.update(pred_np[b][valid].ravel(),
                                label_np[b][valid].ravel())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss    = epoch_loss / len(train_loader)
    train_miou  = train_metric.miou()

    # ── Validation ────────────────────────────────────────────────────────
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
                if args.no_fpn:
                    vis_feat = backbone_out["vision_features"].float()
                else:
                    fpn_out  = fpn(hook_feats)
                    vis_feat = fpn_out["p4"].float()
                vis_up = F.interpolate(
                    vis_feat,
                    size=(args.resolution, args.resolution),
                    mode="bilinear", align_corners=False,
                )
                logits = cls_head(vis_up)

            pred_np  = logits.argmax(dim=1).cpu().numpy()
            label_np = masks.cpu().numpy()
            for b in range(pred_np.shape[0]):
                valid = label_np[b] != 255
                val_metric.update(pred_np[b][valid].ravel(),
                                  label_np[b][valid].ravel())

    val_miou = val_metric.miou()

    # ── 로깅 ──────────────────────────────────────────────────────────────
    print(
        f"Epoch [{epoch+1:>3}/{args.epochs}] "
        f"loss={avg_loss:.4f}  "
        f"train_mIoU={train_miou:.4f}  "
        f"val_mIoU={val_miou:.4f}  "
        f"lr={scheduler.get_last_lr()[0]:.6f}"
    )
    writer.add_scalar("Loss/train",       avg_loss,   epoch + 1)
    writer.add_scalar("mIoU/train",       train_miou, epoch + 1)
    writer.add_scalar("mIoU/val",         val_miou,   epoch + 1)
    writer.add_scalar("LR",               scheduler.get_last_lr()[0], epoch + 1)

    # ── 클래스별 IoU 로깅 (val) ───────────────────────────────────────────
    per_cls_iou = val_metric.per_class_iou()
    print(f"  {'Class':<22} {'IoU':>6}")
    print(f"  {'-'*30}")
    for cls_idx, (name, iou_val) in enumerate(zip(class_names, per_cls_iou)):
        if np.isnan(iou_val):
            print(f"  {name:<22} {'N/A':>6}")
        else:
            print(f"  {name:<22} {iou_val:.4f}")
            writer.add_scalar(f"IoU_val/{name}", iou_val, epoch + 1)

    # ── 체크포인트 저장 ────────────────────────────────────────────────────
    ckpt = {
        "epoch":      epoch + 1,
        "adapters":   adapters.state_dict(),
        "fpn":        fpn.state_dict(),
        "cls_head":   cls_head.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "best_miou":  best_miou,
        "config": {
            "dataset":      args.dataset,
            "num_classes":  num_classes,
            "bottleneck":   args.bottleneck,
            "in_channels":  1024,
            "out_channels": 256,
            "resolution":   args.resolution,
        },
    }

    # 마지막 체크포인트 항상 저장
    torch.save(ckpt, args.save_path)

    # val mIoU 기준 best 모델 별도 저장
    if val_miou > best_miou:
        best_miou = val_miou
        best_path = args.save_path.replace(".pt", "_best.pt")
        torch.save(ckpt, best_path)
        print(f"  ★ Best mIoU 갱신: {best_miou:.4f} → 저장: {best_path}")

# ── 정리 ─────────────────────────────────────────────────────────────────────
writer.close()
for h in hooks:
    h.remove()
print(f"\n학습 완료. 최종 체크포인트: {args.save_path}")
print(f"Best val mIoU: {best_miou:.4f}")
