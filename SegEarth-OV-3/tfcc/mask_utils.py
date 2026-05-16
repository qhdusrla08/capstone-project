from __future__ import annotations

from typing import Any

import numpy as np
import torch


Box = tuple[int, int, int, int]


def mask_iou(mask_a: Any, mask_b: Any) -> float:
    first = _as_bool(mask_a)
    second = _as_bool(mask_b)
    if first.shape != second.shape:
        raise ValueError(f"Mask shape mismatch: {first.shape} vs {second.shape}")
    intersection = np.logical_and(first, second).sum(dtype=np.float64)
    union = np.logical_or(first, second).sum(dtype=np.float64)
    return float(intersection / union) if union > 0 else 1.0


def mask_bbox(mask: Any) -> Box | None:
    binary = _as_bool(mask)
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize_class_name(class_name: str) -> str:
    return " ".join(str(class_name).replace("_", " ").lower().split())


def _as_bool(mask: Any) -> np.ndarray:
    if torch.is_tensor(mask):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {array.shape}")
    return array if array.dtype == bool else array > 0.5
