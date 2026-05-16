from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import RCRConfig, load_rcr_config
from .evidence_extractor import RCREvidenceExtractor, inverse_transform_logits, transform_image


class RCRInferencer:
    """Region-consistency refinement wrapper for SegEarth-OV3 inference."""

    def __init__(
        self,
        base_model: Any,
        class_names: Sequence[str],
        config: RCRConfig | str | Path | None = None,
        output_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.base_model = base_model
        self.class_names = [str(name).strip() for name in class_names if str(name).strip()]
        self.config = load_rcr_config(config) if not isinstance(config, RCRConfig) else config
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = torch.device(device) if device is not None else getattr(base_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.evidence_extractor = RCREvidenceExtractor(
            base_model=base_model,
            class_names=self.class_names,
            device=self.device,
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def infer_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        image_id: str | None = None,
        output_dir: str | Path | None = None,
        save_json: bool | None = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        pil_image = self._load_image(image)
        image_id = image_id or self._image_id(image)
        output_path = Path(output_dir) if output_dir else self.output_dir
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        save_json = self.config.output.save_json if save_json is None else bool(save_json)

        raw = self.evidence_extractor.predict_class_details(pil_image)
        raw_logits = raw["class_logits"].detach()
        raw_seg = self._logits_to_segmentation(raw_logits)
        consensus_logits, consensus_debug = self._compute_consensus_logits(pil_image, raw_logits)
        base_logits = consensus_logits if self.config.tta.use_consensus_as_base else raw_logits
        refined_logits, boundary_debug = self._refine_boundary(base_logits, consensus_logits, raw)
        refined_logits, local_vote_debug = self._refine_local_vote(refined_logits, consensus_logits)
        refined_logits, component_debug = self._cleanup_components(refined_logits, raw_logits)
        refined_logits, safety_debug = self._limit_total_changes(raw_logits, refined_logits)
        refined_seg = self._logits_to_segmentation(refined_logits)

        artifact_paths = self._save_artifacts(image_id, output_path, raw_seg, refined_seg, save_json)
        debug = {
            "runtime_sec": float(time.perf_counter() - start_time),
            "model": "RCR-SegEarth",
            **consensus_debug,
            **boundary_debug,
            **local_vote_debug,
            **component_debug,
            **safety_debug,
        }
        result = {
            "image_id": image_id,
            "segmentation_map": refined_seg.detach().cpu().numpy().astype(np.int32),
            "logits": refined_logits.detach() if self.config.output.keep_logits else None,
            "debug": debug,
            "artifact_paths": artifact_paths,
        }
        if output_path and save_json:
            json_path = output_path / f"{image_id}_rcr.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(_json_safe({k: v for k, v in result.items() if k != "logits"}), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            result["json_path"] = str(json_path)
        return result

    def _compute_consensus_logits(self, image: Image.Image, raw_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.config.tta.enabled:
            return raw_logits.detach().clone(), {"rcr_tta_enabled": False, "rcr_tta_views": []}

        aligned = [raw_logits.detach()]
        used_views: list[str] = []
        for view in self.config.tta.views:
            if view == "orig":
                continue
            transformed = transform_image(image, view)
            details = self.evidence_extractor.predict_class_details(transformed)
            restored = inverse_transform_logits(details["class_logits"], view)
            if restored.shape[-2:] != raw_logits.shape[-2:]:
                restored = F.interpolate(restored.unsqueeze(0), size=raw_logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            aligned.append(restored.detach())
            used_views.append(view)

        if len(aligned) == 1:
            return raw_logits.detach().clone(), {"rcr_tta_enabled": False, "rcr_tta_views": []}
        consensus = torch.stack(aligned, dim=0).mean(dim=0)
        fuse_weight = min(1.0, max(0.0, float(self.config.tta.fuse_weight)))
        fused = (1.0 - fuse_weight) * raw_logits.detach() + fuse_weight * consensus
        return fused.detach(), {
            "rcr_tta_enabled": True,
            "rcr_tta_views": used_views,
            "rcr_tta_fuse_weight": fuse_weight,
            "rcr_tta_use_consensus_as_base": bool(self.config.tta.use_consensus_as_base),
        }

    def _refine_boundary(self, raw_logits: torch.Tensor, consensus_logits: torch.Tensor, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        refined = raw_logits.detach().clone()
        if not self.config.boundary.enabled:
            return refined, {"rcr_boundary_changed_pixels": 0}

        raw_values, raw_labels = raw_logits.max(dim=0)
        raw_top2 = torch.topk(raw_logits, k=min(2, raw_logits.shape[0]), dim=0).values
        raw_margin = raw_top2[0] - raw_top2[1] if raw_top2.shape[0] > 1 else torch.zeros_like(raw_values)
        consensus_values, consensus_labels = consensus_logits.max(dim=0)
        raw_seg = self._logits_to_segmentation(raw_logits)
        boundary = self._boundary_mask(raw_seg, int(self.config.boundary.kernel_size))
        uncertain = raw_margin <= float(self.config.boundary.max_margin)
        high_conf_lock = (raw_values >= float(self.config.boundary.high_confidence_lock)) & (raw_margin >= float(self.config.boundary.lock_margin)) & (~boundary)
        confidence_gate = raw_values >= self._prob_threshold()
        gain_gate = consensus_values >= raw_values + float(self.config.boundary.min_consensus_gain)
        label_gate = consensus_labels != raw_labels
        if not self.config.boundary.allow_background_relabel:
            label_gate = label_gate & (consensus_labels != self._bg_idx())

        candidate = (boundary | uncertain) & (~high_conf_lock) & confidence_gate & gain_gate & label_gate
        if self.config.dense_head.enabled:
            candidate = candidate | self._dense_head_candidate(raw_logits, raw, raw_values, raw_labels)

        candidate = self._cap_mask_by_score(candidate, consensus_values - raw_values, float(self.config.boundary.max_changed_ratio))
        changed_pixels = int(candidate.sum().detach().cpu())
        if changed_pixels > 0:
            target_labels = consensus_labels[candidate]
            old_values = refined.gather(0, raw_labels.unsqueeze(0)).squeeze(0)[candidate]
            refined[:, candidate] = raw_logits[:, candidate]
            refined[target_labels, candidate] = torch.maximum(
                refined[target_labels, candidate],
                old_values + float(self.config.boundary.label_boost),
            )
        return refined, {"rcr_boundary_changed_pixels": changed_pixels}

    def _dense_head_candidate(
        self,
        raw_logits: torch.Tensor,
        raw: dict[str, torch.Tensor],
        raw_values: torch.Tensor,
        raw_labels: torch.Tensor,
    ) -> torch.Tensor:
        semantic = raw["semantic_logits"].to(device=raw_logits.device, dtype=raw_logits.dtype)
        instance = raw["instance_logits"].to(device=raw_logits.device, dtype=raw_logits.dtype)
        consensus = torch.minimum(semantic, instance).clamp_min(0.0)
        disagreement = torch.abs(semantic - instance)
        dense_logits = raw_logits + float(self.config.dense_head.consensus_weight) * consensus - float(self.config.dense_head.disagreement_weight) * disagreement
        dense_values, dense_labels = dense_logits.max(dim=0)
        dense_gain = dense_values >= raw_values + float(self.config.boundary.min_consensus_gain)
        dense_margin = (raw_values - raw_logits.gather(0, dense_labels.unsqueeze(0)).squeeze(0)) <= float(self.config.dense_head.max_margin)
        candidate = (dense_labels != raw_labels) & dense_gain & dense_margin
        if not self.config.boundary.allow_background_relabel:
            candidate = candidate & (dense_labels != self._bg_idx())
        return candidate

    def _cleanup_components(self, logits: torch.Tensor, raw_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.config.component.enabled:
            return logits.detach().clone(), {"rcr_component_changed_pixels": 0, "rcr_component_count": 0}

        refined = logits.detach().clone()
        seg = self._logits_to_segmentation(refined)
        seg_np = seg.detach().cpu().numpy().astype(np.int32)
        image_area = int(seg_np.size)
        min_area = max(int(self.config.component.min_area), int(image_area * float(self.config.component.min_area_ratio)))
        changed = 0
        component_count = 0
        bg_idx = self._bg_idx()
        kernel_size = max(3, int(self.config.component.neighbor_kernel_size))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        raw_values = raw_logits.max(dim=0)[0]

        for class_id in range(refined.shape[0]):
            if class_id == bg_idx:
                continue
            class_mask = (seg_np == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area > min_area:
                    continue
                comp_mask_np = labels == label_id
                comp_mask = torch.as_tensor(comp_mask_np, dtype=torch.bool, device=refined.device)
                if float(raw_values[comp_mask].mean().detach().cpu()) >= float(self.config.component.protect_confidence):
                    continue
                neighbor = self._dominant_neighbor(seg_np, comp_mask_np, class_id, kernel)
                if neighbor is None:
                    continue
                if neighbor == bg_idx and not self.config.component.allow_background_relabel:
                    continue
                old_values = refined[class_id, comp_mask]
                refined[neighbor, comp_mask] = torch.maximum(refined[neighbor, comp_mask], old_values + float(self.config.component.label_boost))
                changed += area
                component_count += 1
        return refined, {"rcr_component_changed_pixels": changed, "rcr_component_count": component_count}

    def _refine_local_vote(self, logits: torch.Tensor, consensus_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.config.local_vote.enabled:
            return logits.detach().clone(), {"rcr_local_vote_changed_pixels": 0}

        refined = logits.detach().clone()
        values, labels = refined.max(dim=0)
        top2 = torch.topk(refined, k=min(2, refined.shape[0]), dim=0).values
        margin = top2[0] - top2[1] if top2.shape[0] > 1 else torch.zeros_like(values)
        seg = self._logits_to_segmentation(refined)
        boundary = self._boundary_mask(seg, int(self.config.boundary.kernel_size))
        vote_labels, vote_ratio = self._local_majority(seg, refined.shape[0], int(self.config.local_vote.kernel_size))
        consensus_values, consensus_labels = consensus_logits.max(dim=0)
        vote_logits = refined.gather(0, vote_labels.unsqueeze(0)).squeeze(0)
        vote_support = vote_logits >= values - float(self.config.local_vote.max_margin)
        consensus_support = consensus_logits.gather(0, vote_labels.unsqueeze(0)).squeeze(0) >= values + float(self.config.local_vote.min_consensus_support)
        if self.config.local_vote.consensus_agree_required:
            consensus_support = consensus_support & (consensus_labels == vote_labels)

        candidate = (
            (boundary | (margin <= float(self.config.local_vote.max_margin)))
            & (vote_labels != labels)
            & (vote_ratio >= float(self.config.local_vote.min_vote_ratio))
            & vote_support
            & consensus_support
            & (values >= self._prob_threshold())
        )
        if not self.config.local_vote.allow_background_relabel:
            candidate = candidate & (vote_labels != self._bg_idx())

        candidate = self._cap_mask_by_score(candidate, vote_ratio + consensus_values - values, float(self.config.local_vote.max_changed_ratio))
        changed_pixels = int(candidate.sum().detach().cpu())
        if changed_pixels > 0:
            target_labels = vote_labels[candidate]
            old_values = refined.gather(0, labels.unsqueeze(0)).squeeze(0)[candidate]
            refined[:, candidate] = logits[:, candidate]
            refined[target_labels, candidate] = torch.maximum(
                refined[target_labels, candidate],
                old_values + float(self.config.local_vote.label_boost),
            )
        return refined, {"rcr_local_vote_changed_pixels": changed_pixels}

    def _local_majority(self, seg: torch.Tensor, num_classes: int, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        valid_seg = seg.clamp(0, num_classes - 1).long()
        one_hot = F.one_hot(valid_seg, num_classes=num_classes).permute(2, 0, 1).float().unsqueeze(0)
        counts = F.avg_pool2d(one_hot, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(0)
        vote_ratio, vote_labels = counts.max(dim=0)
        return vote_labels.long(), vote_ratio

    def _dominant_neighbor(self, seg_np: np.ndarray, component_mask: np.ndarray, class_id: int, kernel: np.ndarray) -> int | None:
        dilated = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & (~component_mask)
        values = seg_np[ring]
        values = values[values != class_id]
        if values.size == 0:
            return None
        labels, counts = np.unique(values, return_counts=True)
        return int(labels[np.argmax(counts)])

    def _limit_total_changes(self, raw_logits: torch.Tensor, refined_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_seg = self._logits_to_segmentation(raw_logits)
        refined_seg = self._logits_to_segmentation(refined_logits)
        changed = raw_seg != refined_seg
        total_count = int(changed.numel())
        changed_count = int(changed.sum().detach().cpu())
        max_ratio = max(0.0, float(self.config.safety.max_total_changed_ratio))
        if changed_count == 0 or changed_count / max(1, total_count) <= max_ratio:
            return refined_logits, {"rcr_total_changed_ratio": changed_count / max(1, total_count), "rcr_safety_limited": False}

        keep_count = max(1, int(total_count * max_ratio))
        raw_values = raw_logits.max(dim=0)[0]
        refined_values = refined_logits.max(dim=0)[0]
        gains = (refined_values - raw_values).detach().flatten()
        changed_flat = changed.flatten()
        changed_indices = torch.nonzero(changed_flat, as_tuple=False).flatten()
        keep_local = torch.topk(gains[changed_indices], k=min(keep_count, changed_indices.numel())).indices
        keep_indices = changed_indices[keep_local]
        keep_flat = torch.zeros_like(changed_flat, dtype=torch.bool)
        keep_flat[keep_indices] = True
        revert_mask = (changed_flat & ~keep_flat).view_as(changed)
        safe = refined_logits.detach().clone()
        safe[:, revert_mask] = raw_logits[:, revert_mask]
        safe_changed = int((self._logits_to_segmentation(raw_logits) != self._logits_to_segmentation(safe)).sum().detach().cpu())
        return safe, {"rcr_total_changed_ratio": safe_changed / max(1, total_count), "rcr_safety_limited": True}

    def _cap_mask_by_score(self, mask: torch.Tensor, score: torch.Tensor, max_ratio: float) -> torch.Tensor:
        if not mask.any():
            return mask
        max_ratio = max(0.0, float(max_ratio))
        if max_ratio <= 0.0:
            return torch.zeros_like(mask)
        max_count = max(1, int(mask.numel() * max_ratio))
        indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        if indices.numel() <= max_count:
            return mask
        scores = score.detach().flatten()[indices]
        keep = indices[torch.topk(scores, k=max_count).indices]
        capped = torch.zeros_like(mask.flatten(), dtype=torch.bool)
        capped[keep] = True
        return capped.view_as(mask)

    def _boundary_mask(self, seg: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        seg_float = seg.float().view(1, 1, *seg.shape)
        max_pool = F.max_pool2d(seg_float, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        min_pool = -F.max_pool2d(-seg_float, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return (max_pool != min_pool).squeeze(0).squeeze(0)

    def _logits_to_segmentation(self, logits: torch.Tensor) -> torch.Tensor:
        seg_pred = torch.argmax(logits, dim=0)
        max_vals = logits.max(dim=0)[0]
        seg_pred[max_vals < self._prob_threshold()] = self._bg_idx()
        return seg_pred

    def _prob_threshold(self) -> float:
        if self.config.prob_threshold is not None:
            return float(self.config.prob_threshold)
        return float(getattr(self.base_model, "prob_thd", 0.0))

    def _bg_idx(self) -> int:
        if self.config.bg_idx is not None:
            return int(self.config.bg_idx)
        return int(getattr(self.base_model, "bg_idx", 0))

    def _save_artifacts(self, image_id: str, output_path: Path | None, raw_seg: torch.Tensor, refined_seg: torch.Tensor, save_enabled: bool) -> dict[str, str]:
        if output_path is None or not save_enabled or not self.config.output.save_masks:
            return {}
        raw_path = output_path / f"{image_id}_rcr_raw_mask.png"
        refined_path = output_path / f"{image_id}_rcr_refined_mask.png"
        Image.fromarray(raw_seg.detach().cpu().numpy().astype(np.uint8)).save(raw_path)
        Image.fromarray(refined_seg.detach().cpu().numpy().astype(np.uint8)).save(refined_path)
        return {"raw_mask_path": str(raw_path), "refined_mask_path": str(refined_path)}

    def _load_image(self, image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

    def _image_id(self, image: str | Path | Image.Image | np.ndarray) -> str:
        if isinstance(image, (str, Path)):
            return Path(image).stem
        return "image"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
