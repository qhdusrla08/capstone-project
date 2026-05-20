#!/usr/bin/env python3
"""Single-image RCR-SegEarth inference + category-filtered overlay.

한 장의 이미지에 대해 임의의 OVSS 클래스 셋으로 추론을 돌리고, 그 중 사용자가
지정한 카테고리만 입력 이미지 위에 반투명 overlay 로 합성해 PNG 로 저장한다.
포스터의 HERO 패널 cell ②③④ 합성용 (text-only OVSS + RCR, box/point indicator 는
figure 편집기에서 수동으로 그려넣는 워크플로우를 가정).

실행 위치: SegEarth-OV-3/ 디렉터리에서 실행해야 한다
(segearthov3_segmentor.py 가 `./sam3/...` 와 `weights/sam3/sam3.pt` 를
상대 경로로 참조하기 때문).

사용 예:
  cd SegEarth-OV-3
  python scripts/single_image_rcr.py \\
      --image resources/oem_koeln_50.tif \\
      --classes background "bareland,barren" grass road "tree,forest" \\
                "water,river" cropland "building,roof,house" solar_panel \\
      --show solar_panel \\
      --out outputs/hero/cell2_text_only.png \\
      --rcr-config configs/rcr_openearthmap.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# segearthov3_segmentor.py 는 프로젝트 루트 (SegEarth-OV-3/) 에 있다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mmseg.structures import SegDataSample  # noqa: E402
from segearthov3_segmentor import SegEarthOV3Segmentation  # noqa: E402


# 시각적으로 구분 잘 되는 16색 (RGB 0–255). class_id 가 이 길이를 넘으면 wrap-around.
DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],         # 0
        [228, 26, 28],     # red
        [55, 126, 184],    # blue
        [77, 175, 74],     # green
        [152, 78, 163],    # purple
        [255, 127, 0],     # orange
        [255, 255, 51],    # yellow
        [166, 86, 40],     # brown
        [247, 129, 191],   # pink
        [153, 153, 153],   # grey
        [102, 194, 165],   # teal
        [252, 141, 98],    # salmon
        [141, 160, 203],   # lavender
        [231, 138, 195],   # rose
        [166, 216, 84],    # lime
        [255, 217, 47],    # gold
    ],
    dtype=np.uint8,
)


def resolve_show_indices(classes: list[str], show: list[str]) -> list[int]:
    """`--show` 의 이름들을 클래스 인덱스로 매핑한다.

    case-insensitive. 각 class entry 의 동의어(콤마 분리) 중 하나라도
    매칭되면 그 entry 의 idx 를 반환.
    """
    show_lower = {s.strip().lower() for s in show}
    indices: list[int] = []
    for idx, cls_entry in enumerate(classes):
        synonyms = {s.strip().lower() for s in cls_entry.split(",")}
        if synonyms & show_lower:
            indices.append(idx)
    return indices


def overlay_on_image(
    rgb: np.ndarray,
    seg: np.ndarray,
    show_indices: list[int],
    palette: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """show_indices 에 속한 픽셀만 컬러 overlay. 나머지는 원본 이미지 그대로."""
    out = rgb.astype(np.float32).copy()
    show_mask = np.zeros(seg.shape, dtype=bool)
    for idx in show_indices:
        cls_mask = seg == idx
        if not cls_mask.any():
            continue
        color = palette[idx % len(palette)].astype(np.float32)
        for c in range(3):
            out[..., c][cls_mask] = (
                (1.0 - alpha) * out[..., c][cls_mask] + alpha * color[c]
            )
        show_mask |= cls_mask
    return out.clip(0, 255).astype(np.uint8), show_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--classes",
        nargs="+",
        help="클래스 엔트리 (1엔트리당 1 arg, 동의어는 콤마 구분)",
    )
    grp.add_argument(
        "--classes-file",
        help="텍스트 파일 (한 줄에 1엔트리, 동의어 콤마 구분)",
    )
    p.add_argument(
        "--show",
        nargs="+",
        required=True,
        help="overlay 로 그릴 클래스 이름 (나머지는 원본 이미지 그대로)",
    )
    p.add_argument("--out", required=True, help="출력 PNG 경로")
    p.add_argument(
        "--rcr-config",
        default="configs/rcr_openearthmap.yaml",
        help="RCR YAML 설정 파일 경로",
    )
    p.add_argument("--bg-idx", type=int, default=0)
    p.add_argument("--prob-thd", type=float, default=0.1)
    p.add_argument("--confidence-threshold", type=float, default=0.1)
    p.add_argument("--slide-crop", type=int, default=512)
    p.add_argument("--slide-stride", type=int, default=512)
    p.add_argument(
        "--alpha", type=float, default=0.5, help="overlay alpha (0..1)"
    )
    p.add_argument(
        "--no-rcr",
        action="store_true",
        help="RCR 비활성화 (baseline SegEarth-OV-3 만)",
    )
    p.add_argument(
        "--save-class-map",
        action="store_true",
        help="추론된 정수 class map 도 함께 저장 (디버깅용)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="실행 디바이스 (기본: cuda)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 클래스 리스트 로딩
    if args.classes:
        classes = [c.strip() for c in args.classes]
    else:
        with open(args.classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]

    show_indices = resolve_show_indices(classes, args.show)
    if not show_indices:
        sys.exit(
            f"[error] --show {args.show} 가 클래스 목록 {classes} 의 어떤 엔트리와도 매칭되지 않음"
        )

    # 2. 임시 classname 파일 작성 (segmentor 는 파일 경로를 요구)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(classes))
        classname_path = f.name

    # 3. Segmentor 빌드 (use_rcr 기본 ON, --no-rcr 로만 끔)
    model = SegEarthOV3Segmentation(
        type="SegEarthOV3Segmentation",
        model_type="SAM3",
        classname_path=classname_path,
        prob_thd=args.prob_thd,
        confidence_threshold=args.confidence_threshold,
        slide_stride=args.slide_stride,
        slide_crop=args.slide_crop,
        bg_idx=args.bg_idx,
        use_rcr=(not args.no_rcr),
        rcr_config_path=args.rcr_config,
    )

    # 4. 이미지 + meta 준비
    img = Image.open(args.image).convert("RGB")
    img_np = np.asarray(img)
    img_tensor = (
        transforms.ToTensor()(img).unsqueeze(0).to(args.device)
    )
    data_sample = SegDataSample()
    data_sample.set_metainfo(
        {"img_path": args.image, "ori_shape": img.size[::-1]}
    )

    # 5. 추론
    seg_pred = model.predict(img_tensor, data_samples=[data_sample])
    seg = (
        seg_pred[0]
        .pred_sem_seg.data.cpu()
        .numpy()
        .squeeze(0)
        .astype(np.int32)
    )

    # 6. 지정 카테고리만 overlay
    overlay_img, show_mask = overlay_on_image(
        img_np, seg, show_indices, DEFAULT_PALETTE, alpha=args.alpha
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    Image.fromarray(overlay_img).save(args.out)

    matched_names = [classes[i] for i in show_indices]
    px_total = int(show_mask.size)
    px_shown = int(show_mask.sum())
    print(f"[ok] saved overlay -> {args.out}")
    print(f"     RCR: {'OFF (baseline)' if args.no_rcr else 'ON'} ({args.rcr_config})")
    print(f"     shown classes: {matched_names} (idx={show_indices})")
    print(
        f"     pixel coverage: {px_shown}/{px_total} "
        f"({100.0 * px_shown / max(px_total, 1):.2f}%)"
    )

    if args.save_class_map:
        cm_path = os.path.splitext(args.out)[0] + "_classmap.png"
        Image.fromarray(seg.astype(np.uint8)).save(cm_path)
        print(f"[ok] saved class map -> {cm_path}")


if __name__ == "__main__":
    main()
