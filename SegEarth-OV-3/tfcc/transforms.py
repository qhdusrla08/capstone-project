from __future__ import annotations

import torch
from PIL import Image


def transform_image(image: Image.Image, view: str) -> Image.Image:
    if view == "orig":
        return image.copy()
    if view == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if view == "rot90":
        return image.rotate(90, expand=True)
    if view == "rot180":
        return image.rotate(180, expand=True)
    if view == "rot270":
        return image.rotate(270, expand=True)
    raise ValueError(f"Unsupported TFCC view: {view}")


def inverse_transform_logits(logits: torch.Tensor, view: str) -> torch.Tensor:
    if view == "orig":
        return logits
    if view == "hflip":
        return torch.flip(logits, dims=(-1,))
    if view == "rot90":
        return torch.rot90(logits, k=-1, dims=(-2, -1))
    if view == "rot180":
        return torch.rot90(logits, k=2, dims=(-2, -1))
    if view == "rot270":
        return torch.rot90(logits, k=1, dims=(-2, -1))
    raise ValueError(f"Unsupported TFCC view: {view}")
