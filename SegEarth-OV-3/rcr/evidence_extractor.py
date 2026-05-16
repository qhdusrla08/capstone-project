from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class RCREvidenceExtractor:
    """Extract SegEarth-OV3 class evidence needed by RCR."""

    def __init__(
        self,
        base_model: Any,
        class_names: Sequence[str],
        device: str | torch.device | None = None,
    ) -> None:
        self.base_model = base_model
        self.class_names = [str(name).strip() for name in class_names if str(name).strip()]
        self.device = torch.device(device) if device is not None else getattr(base_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def predict_class_details(self, image: Image.Image, class_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if getattr(self.base_model, "slide_crop", 0) > 0 and (
            self.base_model.slide_crop < image.size[0] or self.base_model.slide_crop < image.size[1]
        ):
            return self._predict_slide_details(image, class_ids)
        return self._predict_single_view_details(image, class_ids)

    def _predict_single_view_details(self, image: Image.Image, class_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        width, height = image.size
        query_indices = self._query_indices_for_classes(class_ids)
        num_queries = int(getattr(self.base_model, "num_queries", len(self.base_model.query_words)))
        query_logits = torch.zeros((num_queries, height, width), device=self.device)
        semantic_logits = torch.zeros_like(query_logits)
        instance_logits = torch.zeros_like(query_logits)
        semantic_raw_logits = torch.zeros_like(query_logits)
        instance_raw_logits = torch.zeros_like(query_logits)
        presence = torch.zeros((num_queries,), device=self.device)

        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() and self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_ctx:
            state = self.base_model.processor.set_image(image)
            for query_index in query_indices:
                query_word = self.base_model.query_words[query_index]
                self.base_model.processor.reset_all_prompts(state)
                state = self.base_model.processor.set_text_prompt(state=state, prompt=query_word)
                sem_logit = self._extract_semantic_logit(state, height, width)
                inst_logit = self._extract_instance_logit(state, height, width)
                sem_raw_logit = sem_logit.detach().clone()
                inst_raw_logit = inst_logit.detach().clone()
                query_logit = torch.maximum(sem_logit, inst_logit)
                presence_score = state.get("presence_score", torch.tensor(1.0, device=self.device))
                presence_value = presence_score.detach().float().to(self.device).reshape(-1)[0]
                if getattr(self.base_model, "use_presence_score", True):
                    query_logit = query_logit * presence_value
                    sem_logit = sem_logit * presence_value
                    inst_logit = inst_logit * presence_value
                query_logits[query_index] = query_logit
                semantic_logits[query_index] = sem_logit
                instance_logits[query_index] = inst_logit
                semantic_raw_logits[query_index] = sem_raw_logit
                instance_raw_logits[query_index] = inst_raw_logit
                presence[query_index] = presence_value
            self.base_model.processor.reset_all_prompts(state)

        return self._query_details_to_class_details(
            query_logits,
            semantic_logits,
            instance_logits,
            semantic_raw_logits,
            instance_raw_logits,
            presence,
        )

    def _predict_slide_details(self, image: Image.Image, class_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        width, height = image.size
        stride = getattr(self.base_model, "slide_stride", 0)
        crop_size = getattr(self.base_model, "slide_crop", 0)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        num_cls = int(getattr(self.base_model, "num_cls", len(self.class_names)))
        sums = {
            "class_logits": torch.zeros((num_cls, height, width), device=self.device),
            "semantic_logits": torch.zeros((num_cls, height, width), device=self.device),
            "instance_logits": torch.zeros((num_cls, height, width), device=self.device),
            "semantic_raw_logits": torch.zeros((num_cls, height, width), device=self.device),
            "instance_raw_logits": torch.zeros((num_cls, height, width), device=self.device),
        }
        presence_sum = torch.zeros((num_cls,), device=self.device)
        counts = torch.zeros((1, height, width), device=self.device)
        tile_count = 0
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, height)
                x2 = min(x1 + w_crop, width)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_details = self._predict_single_view_details(image.crop((x1, y1, x2, y2)), class_ids)
                for key in sums:
                    sums[key][:, y1:y2, x1:x2] += crop_details[key]
                presence_sum += crop_details["presence"]
                counts[:, y1:y2, x1:x2] += 1
                tile_count += 1

        for key in sums:
            sums[key] = sums[key] / counts.clamp_min(1.0)
        sums["presence"] = presence_sum / max(1, tile_count)
        sums["head_agreement"] = self._compute_head_agreement(sums["semantic_raw_logits"], sums["instance_raw_logits"])
        sums["mask_scores"] = sums["class_logits"].flatten(1).max(dim=1)[0]
        return sums

    def _extract_semantic_logit(self, state: dict[str, Any], height: int, width: int) -> torch.Tensor:
        semantic = state.get("semantic_mask_logits")
        if not torch.is_tensor(semantic):
            return torch.zeros((height, width), device=self.device)
        semantic = semantic.squeeze()
        if semantic.shape != (height, width):
            semantic = F.interpolate(semantic.view(1, 1, *semantic.shape), size=(height, width), mode="bilinear", align_corners=False).squeeze()
        return semantic.float()

    def _extract_instance_logit(self, state: dict[str, Any], height: int, width: int) -> torch.Tensor:
        masks = state.get("masks_logits")
        scores = state.get("object_score")
        if not torch.is_tensor(masks) or masks.numel() == 0 or not torch.is_tensor(scores):
            return torch.zeros((height, width), device=self.device)
        candidates: list[torch.Tensor] = []
        for inst_id in range(masks.shape[0]):
            instance = masks[inst_id].squeeze()
            if instance.shape != (height, width):
                instance = F.interpolate(instance.view(1, 1, *instance.shape), size=(height, width), mode="bilinear", align_corners=False).squeeze()
            candidates.append(instance.float() * scores[inst_id].float())
        return torch.stack(candidates, dim=0).max(dim=0)[0] if candidates else torch.zeros((height, width), device=self.device)

    def _query_details_to_class_details(
        self,
        query_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
        instance_logits: torch.Tensor,
        semantic_raw_logits: torch.Tensor,
        instance_raw_logits: torch.Tensor,
        query_presence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_cls = int(getattr(self.base_model, "num_cls", len(self.class_names)))
        query_idx = getattr(self.base_model, "query_idx")
        if not torch.is_tensor(query_idx):
            query_idx = torch.as_tensor(query_idx, dtype=torch.long, device=query_logits.device)
        class_logits = torch.zeros((num_cls, *query_logits.shape[-2:]), device=self.device)
        class_semantic = torch.zeros_like(class_logits)
        class_instance = torch.zeros_like(class_logits)
        class_semantic_raw = torch.zeros_like(class_logits)
        class_instance_raw = torch.zeros_like(class_logits)
        class_presence = torch.zeros((num_cls,), device=self.device)
        for class_id in range(num_cls):
            alias_mask = query_idx.to(query_logits.device) == class_id
            if alias_mask.any():
                class_logits[class_id] = query_logits[alias_mask].max(dim=0)[0]
                class_semantic[class_id] = semantic_logits[alias_mask].max(dim=0)[0]
                class_instance[class_id] = instance_logits[alias_mask].max(dim=0)[0]
                class_semantic_raw[class_id] = semantic_raw_logits[alias_mask].max(dim=0)[0]
                class_instance_raw[class_id] = instance_raw_logits[alias_mask].max(dim=0)[0]
                class_presence[class_id] = query_presence[alias_mask].max()
        return {
            "class_logits": class_logits.detach(),
            "semantic_logits": class_semantic.detach(),
            "instance_logits": class_instance.detach(),
            "semantic_raw_logits": class_semantic_raw.detach(),
            "instance_raw_logits": class_instance_raw.detach(),
            "presence": class_presence.detach(),
            "head_agreement": self._compute_head_agreement(class_semantic_raw, class_instance_raw),
            "mask_scores": class_logits.flatten(1).max(dim=1)[0].detach(),
        }

    def _compute_head_agreement(self, semantic_logits: torch.Tensor, instance_logits: torch.Tensor) -> torch.Tensor:
        threshold = self._prob_threshold()
        scores = []
        for class_id in range(semantic_logits.shape[0]):
            sem_mask = semantic_logits[class_id] >= threshold
            inst_mask = instance_logits[class_id] >= threshold
            if not sem_mask.any() and not inst_mask.any():
                scores.append(0.0)
            else:
                scores.append(_mask_iou(sem_mask, inst_mask))
        return torch.as_tensor(scores, dtype=torch.float32, device=self.device)

    def _query_indices_for_classes(self, class_ids: Sequence[int] | None) -> list[int]:
        if class_ids is None:
            return list(range(int(getattr(self.base_model, "num_queries", len(self.base_model.query_words)))))
        query_idx = getattr(self.base_model, "query_idx")
        if torch.is_tensor(query_idx):
            query_idx = query_idx.detach().cpu().tolist()
        requested = {int(class_id) for class_id in class_ids}
        return [index for index, class_id in enumerate(query_idx) if int(class_id) in requested]

    def _prob_threshold(self) -> float:
        return float(getattr(self.base_model, "prob_thd", 0.0))


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
    raise ValueError(f"Unsupported RCR view: {view}")


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
    raise ValueError(f"Unsupported RCR view: {view}")


def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    first = mask_a.detach().bool().cpu().numpy()
    second = mask_b.detach().bool().cpu().numpy()
    intersection = np.logical_and(first, second).sum(dtype=np.float64)
    union = np.logical_or(first, second).sum(dtype=np.float64)
    return float(intersection / union) if union > 0 else 1.0
