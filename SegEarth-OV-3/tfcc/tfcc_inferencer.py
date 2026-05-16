from __future__ import annotations

from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import TFCCConfig, load_tfcc_config
from .mask_utils import mask_bbox, mask_iou, normalize_class_name
from .transforms import inverse_transform_logits, transform_image


class TFCCInferencer:
    """Final-fusion confidence calibration for SegEarth-OV3 without weight updates."""

    def __init__(
        self,
        base_model: Any,
        class_names: Sequence[str],
        config: TFCCConfig | str | Path | None = None,
        output_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.base_model = base_model
        self.class_names = [str(name).strip() for name in class_names if str(name).strip()]
        self.config = load_tfcc_config(config) if not isinstance(config, TFCCConfig) else config
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = torch.device(device) if device is not None else getattr(base_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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

        raw = self._predict_class_details(pil_image)
        raw_logits = raw["class_logits"]
        head_agree = raw["head_agreement"]
        presence = raw["presence"]
        mask_scores = raw["mask_scores"]
        raw_seg = self._logits_to_segmentation(raw_logits)

        working_logits, tta_target_logits, tta_debug = self._compute_tta_logits(pil_image, raw_logits)
        working_mask_scores = working_logits.flatten(1).max(dim=1)[0].detach()
        working_logits, online_debug = self._optimize_online_logits(
            base_logits=working_logits,
            target_logits=tta_target_logits,
            presence=presence,
            mask_scores=working_mask_scores,
        )
        working_mask_scores = working_logits.flatten(1).max(dim=1)[0].detach()
        selected_geo_classes = self._select_geo_classes(presence, working_mask_scores)
        geo_scores = self._compute_geo_scores(pil_image, working_logits, selected_geo_classes)
        calibrated_logits, calibration_debug = self._calibrate_logits(working_logits, head_agree, geo_scores, presence, working_mask_scores, raw)
        calibrated_seg = self._logits_to_segmentation(calibrated_logits)
        exemplar_bank = self._build_exemplar_bank(raw, raw_seg)
        artifact_paths = self._save_artifacts(
            image_id=image_id,
            output_path=output_path,
            raw=raw,
            raw_seg=raw_seg,
            calibrated_logits=calibrated_logits,
            calibrated_seg=calibrated_seg,
            save_enabled=save_json,
        )

        debug = {
            "runtime_sec": float(time.perf_counter() - start_time),
            "num_geo_classes": len(selected_geo_classes),
            "geo_classes": [self.class_names[class_id] for class_id in selected_geo_classes if class_id < len(self.class_names)],
            "head_agreement": _tensor_to_float_list(head_agree),
            "geo_scores": _tensor_to_float_list(geo_scores),
            "presence": _tensor_to_float_list(presence),
            "num_exemplars": len(exemplar_bank),
            "calibration_mode": "zero-shot TFCC-SegEarth" if self.config.zero_shot_mode else "validation-calibrated TFCC-SegEarth",
        }
        debug.update(tta_debug)
        debug.update(online_debug)
        debug.update(calibration_debug)
        result = {
            "image_id": image_id,
            "segmentation_map": calibrated_seg.detach().cpu().numpy().astype(np.int32),
            "logits": calibrated_logits.detach() if self.config.output.keep_logits else None,
            "exemplar_bank": exemplar_bank,
            "debug": debug,
        }
        if self.config.output.save_raw_result:
            raw_result = {
                "final_mask": raw_seg.detach().cpu().numpy().astype(np.int32),
                "mask_scores": _tensor_to_float_list(mask_scores),
                "presence": _tensor_to_float_list(presence),
                "head_agreement": _tensor_to_float_list(head_agree),
                "semantic_mask": (raw["semantic_raw_logits"] >= self._mask_threshold()).detach().cpu().numpy().astype(np.uint8),
                "instance_mask": (raw["instance_raw_logits"] >= self._mask_threshold()).detach().cpu().numpy().astype(np.uint8),
                "artifact_paths": artifact_paths.get("raw_result", {}),
            }
            if self.config.output.keep_dense_results:
                raw_result["raw_logit"] = raw_logits.detach()
                raw_result["semantic_score_map"] = raw["semantic_raw_logits"].detach()
                raw_result["instance_score_map"] = raw["instance_raw_logits"].detach()
            result["raw_result"] = raw_result
        if self.config.output.save_calibrated_result:
            calibrated_result = {
                "final_mask": calibrated_seg.detach().cpu().numpy().astype(np.int32),
                "geo_scores": _tensor_to_float_list(geo_scores),
                "artifact_paths": artifact_paths.get("calibrated_result", {}),
            }
            if self.config.output.keep_dense_results:
                calibrated_result["calibrated_logit"] = calibrated_logits.detach()
            result["calibrated_result"] = calibrated_result
        if output_path and save_json:
            json_path = output_path / f"{image_id}_tfcc.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(self._json_payload(result), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            result["json_path"] = str(json_path)
        return result

    def _predict_class_details(self, image: Image.Image, class_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
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
        threshold = self._mask_threshold()
        scores = []
        for class_id in range(semantic_logits.shape[0]):
            sem_mask = semantic_logits[class_id] >= threshold
            inst_mask = instance_logits[class_id] >= threshold
            if not sem_mask.any() and not inst_mask.any():
                scores.append(0.0)
                continue
            try:
                scores.append(mask_iou(sem_mask, inst_mask))
            except ValueError:
                scores.append(0.0)
        return torch.as_tensor(scores, dtype=torch.float32, device=self.device)

    def _select_geo_classes(self, presence: torch.Tensor, mask_scores: torch.Tensor) -> list[int]:
        if not self.config.geometry.enabled:
            return []
        excluded = {normalize_class_name(item) for item in self.config.exclude_classes}
        valid_ids = [
            class_id
            for class_id, class_name in enumerate(self.class_names)
            if normalize_class_name(class_name.split(",", 1)[0]) not in excluded
        ]
        if not valid_ids:
            return []
        top_k = min(max(0, int(self.config.geometry.presence_top_k)), len(valid_ids))
        valid_tensor = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
        top_indices = presence[valid_tensor].topk(k=top_k).indices if top_k > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        top_classes = valid_tensor[top_indices].detach().cpu().tolist()
        selected = []
        for class_id in top_classes:
            score = float(mask_scores[class_id].detach().cpu())
            if float(self.config.thresholds.uncertain_low) <= score <= float(self.config.thresholds.uncertain_high):
                selected.append(int(class_id))
        return selected[: max(0, int(self.config.geometry.max_geo_classes))]

    def _compute_geo_scores(self, image: Image.Image, raw_logits: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
        geo_scores = torch.zeros((raw_logits.shape[0],), dtype=torch.float32, device=self.device)
        if not class_ids:
            return geo_scores
        threshold = self._mask_threshold()
        original_probs = raw_logits[list(class_ids)].detach()
        aligned_views = [original_probs]
        for view in self.config.geometry.views:
            transformed = transform_image(image, view)
            details = self._predict_class_details(transformed, class_ids=class_ids)
            view_logits = details["class_logits"][list(class_ids)]
            restored = inverse_transform_logits(view_logits, view)
            if restored.shape[-2:] != raw_logits.shape[-2:]:
                restored = F.interpolate(restored.unsqueeze(0), size=raw_logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            aligned_views.append(restored.detach())
        stacked = torch.stack(aligned_views, dim=0)
        variance_score = (1.0 - stacked.var(dim=0).flatten(1).mean(dim=1)).clamp(0.0, 1.0)
        iou_scores = []
        original_masks = original_probs >= threshold
        for idx, _class_id in enumerate(class_ids):
            view_ious = []
            for view_idx in range(1, stacked.shape[0]):
                view_ious.append(mask_iou(original_masks[idx], stacked[view_idx, idx] >= threshold))
            iou_scores.append(float(sum(view_ious) / max(1, len(view_ious))))
        iou_tensor = torch.as_tensor(iou_scores, dtype=torch.float32, device=self.device)
        variance_weight = float(self.config.geometry.variance_weight)
        selected_scores = ((1.0 - variance_weight) * iou_tensor + variance_weight * variance_score).clamp(0.0, 1.0)
        for idx, class_id in enumerate(class_ids):
            geo_scores[int(class_id)] = selected_scores[idx]
        return geo_scores

    def _compute_tta_logits(self, image: Image.Image, raw_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if not self.config.tta.enabled:
            return raw_logits.detach().clone(), None, {
                "tta_enabled": False,
                "tta_views": [],
                "tta_fuse_weight": 0.0,
            }

        aligned_views = [raw_logits.detach()]
        used_views: list[str] = []
        for view in self.config.tta.views:
            if view == "orig":
                continue
            transformed = transform_image(image, view)
            details = self._predict_class_details(transformed)
            restored = inverse_transform_logits(details["class_logits"], view)
            if restored.shape[-2:] != raw_logits.shape[-2:]:
                restored = F.interpolate(restored.unsqueeze(0), size=raw_logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            aligned_views.append(restored.detach())
            used_views.append(view)

        if len(aligned_views) == 1:
            return raw_logits.detach().clone(), None, {
                "tta_enabled": False,
                "tta_views": [],
                "tta_fuse_weight": 0.0,
            }

        tta_logits = torch.stack(aligned_views, dim=0).mean(dim=0)
        fuse_weight = min(1.0, max(0.0, float(self.config.tta.fuse_weight)))
        fused = (1.0 - fuse_weight) * raw_logits.detach() + fuse_weight * tta_logits
        return fused.detach(), tta_logits.detach(), {
            "tta_enabled": True,
            "tta_views": used_views,
            "tta_fuse_weight": fuse_weight,
        }

    def _optimize_online_logits(
        self,
        base_logits: torch.Tensor,
        target_logits: torch.Tensor | None,
        presence: torch.Tensor,
        mask_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.config.online.enabled or target_logits is None:
            return base_logits.detach().clone(), {
                "online_enabled": False,
                "online_steps": 0,
                "online_classes": [],
                "online_changed_ratio": 0.0,
            }

        selected_classes = self._select_online_classes(presence, mask_scores)
        pixel_gate = self._online_pixel_gate(base_logits, target_logits, selected_classes)
        if not selected_classes or not pixel_gate.any():
            return base_logits.detach().clone(), {
                "online_enabled": False,
                "online_steps": 0,
                "online_classes": [self.class_names[class_id] for class_id in selected_classes if class_id < len(self.class_names)],
                "online_changed_ratio": 0.0,
            }

        class_mask = torch.zeros((base_logits.shape[0],), dtype=base_logits.dtype, device=base_logits.device)
        class_mask[selected_classes] = 1.0
        raw_base = base_logits.detach()
        target = target_logits.detach()
        temperature = max(1.0e-4, float(self.config.online.temperature))
        max_abs_bias = max(0.0, float(self.config.online.max_abs_bias))
        bias_param = torch.zeros((base_logits.shape[0],), dtype=torch.float32, device=base_logits.device, requires_grad=True)
        optimizer = torch.optim.AdamW(
            [bias_param],
            lr=max(0.0, float(self.config.online.lr)),
            weight_decay=max(0.0, float(self.config.online.weight_decay)),
        )
        target_probs = F.softmax(target[:, pixel_gate] / temperature, dim=0).detach()
        loss_value = 0.0
        with torch.enable_grad():
            for _step in range(max(0, int(self.config.online.steps))):
                optimizer.zero_grad(set_to_none=True)
                bounded_bias = torch.tanh(bias_param) * max_abs_bias * class_mask
                logits = raw_base[:, pixel_gate] + bounded_bias.view(-1, 1)
                log_probs = F.log_softmax(logits / temperature, dim=0)
                probs = log_probs.exp()
                kl_loss = F.kl_div(log_probs.transpose(0, 1), target_probs.transpose(0, 1), reduction="batchmean")
                entropy = -(probs * log_probs).sum(dim=0).mean()
                bias_l2 = bounded_bias.pow(2).mean()
                loss = (
                    kl_loss
                    + float(self.config.online.entropy_weight) * entropy
                    + float(self.config.online.bias_l2_weight) * bias_l2
                )
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu())

        with torch.no_grad():
            final_bias = torch.tanh(bias_param.detach()) * max_abs_bias * class_mask
            optimized = raw_base + final_bias.view(-1, 1, 1)
            optimized, changed_ratio, limited = self._limit_changed_pixels(raw_base, optimized)
        return optimized.detach(), {
            "online_enabled": True,
            "online_steps": max(0, int(self.config.online.steps)),
            "online_classes": [self.class_names[class_id] for class_id in selected_classes if class_id < len(self.class_names)],
            "online_class_ids": selected_classes,
            "online_pixel_count": int(pixel_gate.sum().detach().cpu()),
            "online_changed_ratio": changed_ratio,
            "online_limited": limited,
            "online_loss": loss_value,
            "online_bias": _tensor_to_float_list(final_bias),
        }

    def _select_online_classes(self, presence: torch.Tensor, mask_scores: torch.Tensor) -> list[int]:
        excluded = {normalize_class_name(item) for item in self.config.exclude_classes}
        valid_ids = [
            class_id
            for class_id, class_name in enumerate(self.class_names)
            if normalize_class_name(class_name.split(",", 1)[0]) not in excluded
        ]
        if not valid_ids:
            return []
        top_k = min(max(0, int(self.config.online.presence_top_k)), len(valid_ids))
        valid_tensor = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
        top_indices = presence[valid_tensor].topk(k=top_k).indices if top_k > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        top_classes = valid_tensor[top_indices].detach().cpu().tolist()
        uncertain_low = float(self.config.thresholds.uncertain_low)
        uncertain_high = float(self.config.thresholds.uncertain_high)
        selected = []
        for class_id in top_classes:
            score = float(mask_scores[class_id].detach().cpu())
            if uncertain_low <= score <= uncertain_high:
                selected.append(int(class_id))
        if not selected:
            selected = [int(class_id) for class_id in top_classes]
        return selected[: max(0, int(self.config.online.max_classes))]

    def _online_pixel_gate(self, base_logits: torch.Tensor, target_logits: torch.Tensor, selected_classes: Sequence[int]) -> torch.Tensor:
        if not selected_classes:
            return torch.zeros(base_logits.shape[-2:], dtype=torch.bool, device=base_logits.device)
        selected_mask = torch.zeros((base_logits.shape[0],), dtype=torch.bool, device=base_logits.device)
        selected_mask[list(selected_classes)] = True
        base_values, base_labels = base_logits.max(dim=0)
        target_labels = target_logits.argmax(dim=0)
        top2 = torch.topk(base_logits, k=min(2, base_logits.shape[0]), dim=0).values
        if top2.shape[0] > 1:
            margin = top2[0] - top2[1]
        else:
            margin = torch.zeros_like(base_values)
        min_confidence = self._prob_threshold() if self.config.online.min_confidence is None else float(self.config.online.min_confidence)
        label_gate = selected_mask[base_labels] | selected_mask[target_labels]
        uncertainty_gate = (margin <= max(0.0, float(self.config.online.pixel_margin))) | (base_labels != target_labels)
        confidence_gate = base_values >= min_confidence
        return label_gate & uncertainty_gate & confidence_gate

    def _calibrate_logits(
        self,
        raw_logits: torch.Tensor,
        head_agree: torch.Tensor,
        geo_scores: torch.Tensor,
        presence: torch.Tensor,
        mask_scores: torch.Tensor,
        raw: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        calibrated = raw_logits.detach().clone()
        if not self.config.calibration.enabled:
            return calibrated, {
                "calibrated_classes": [],
                "calibration_changed_ratio": 0.0,
                "calibration_limited": False,
            }

        selected_classes = self._select_calibration_classes(presence, mask_scores, geo_scores)
        if not selected_classes:
            return calibrated, {
                "calibrated_classes": [],
                "calibration_changed_ratio": 0.0,
                "calibration_limited": False,
            }

        head_weight = float(self.config.weights.head_agreement)
        geo_weight = float(self.config.weights.geometric)
        support_threshold = self._calibration_support_threshold()
        max_margin = max(0.0, float(self.config.calibration.max_margin))
        raw_top_values, raw_top_labels = raw_logits.max(dim=0)
        changed_candidate_pixels = 0

        for class_id in selected_classes:
            if self.config.dense_head.enabled:
                dense_gate = self._apply_dense_head_consensus(
                    calibrated=calibrated,
                    class_id=class_id,
                    raw=raw,
                    raw_top_values=raw_top_values,
                    raw_top_labels=raw_top_labels,
                    support_threshold=support_threshold,
                    max_margin=max_margin,
                )
                changed_candidate_pixels += dense_gate

            delta = head_weight * float(head_agree[class_id].detach().cpu()) + geo_weight * float(geo_scores[class_id].detach().cpu())
            if delta <= 0.0:
                continue
            class_logits = raw_logits[class_id]
            margin_to_winner = raw_top_values - class_logits
            pixel_gate = (class_logits >= support_threshold) & ((margin_to_winner <= max_margin) | (raw_top_labels == class_id))
            changed_candidate_pixels += int(pixel_gate.sum().detach().cpu())
            if pixel_gate.any():
                calibrated[class_id, pixel_gate] += torch.as_tensor(delta, dtype=calibrated.dtype, device=calibrated.device)

        calibrated, changed_ratio, limited = self._limit_changed_pixels(raw_logits, calibrated)
        return calibrated, {
            "calibrated_classes": [self.class_names[class_id] for class_id in selected_classes if class_id < len(self.class_names)],
            "calibrated_class_ids": selected_classes,
            "calibration_candidate_pixels": changed_candidate_pixels,
            "calibration_changed_ratio": changed_ratio,
            "calibration_limited": limited,
            "calibration_support_threshold": support_threshold,
            "calibration_max_margin": max_margin,
            "dense_head_enabled": bool(self.config.dense_head.enabled),
        }

    def _apply_dense_head_consensus(
        self,
        calibrated: torch.Tensor,
        class_id: int,
        raw: dict[str, torch.Tensor],
        raw_top_values: torch.Tensor,
        raw_top_labels: torch.Tensor,
        support_threshold: float,
        max_margin: float,
    ) -> int:
        semantic = raw["semantic_logits"][class_id].to(device=calibrated.device, dtype=calibrated.dtype)
        instance = raw["instance_logits"][class_id].to(device=calibrated.device, dtype=calibrated.dtype)
        class_logits = calibrated[class_id]
        margin_to_winner = raw_top_values - class_logits
        pixel_gate = (class_logits >= support_threshold) & ((margin_to_winner <= max_margin) | (raw_top_labels == class_id))
        if not pixel_gate.any():
            return 0

        consensus = torch.minimum(semantic, instance).clamp_min(0.0)
        disagreement = torch.abs(semantic - instance)
        dense_delta = (
            float(self.config.dense_head.consensus_weight) * consensus
            - float(self.config.dense_head.disagreement_weight) * disagreement
        )
        calibrated[class_id, pixel_gate] += dense_delta[pixel_gate]
        return int(pixel_gate.sum().detach().cpu())

    def _select_calibration_classes(self, presence: torch.Tensor, mask_scores: torch.Tensor, geo_scores: torch.Tensor) -> list[int]:
        excluded = {normalize_class_name(item) for item in self.config.exclude_classes}
        valid_ids = [
            class_id
            for class_id, class_name in enumerate(self.class_names)
            if normalize_class_name(class_name.split(",", 1)[0]) not in excluded
        ]
        if not valid_ids:
            return []

        top_k = min(max(0, int(self.config.calibration.presence_top_k)), len(valid_ids))
        valid_tensor = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
        top_indices = presence[valid_tensor].topk(k=top_k).indices if top_k > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        top_classes = valid_tensor[top_indices].detach().cpu().tolist()
        selected: list[int] = []
        uncertain_low = float(self.config.thresholds.uncertain_low)
        uncertain_high = float(self.config.thresholds.uncertain_high)
        geo_class_ids = {int(class_id) for class_id in torch.nonzero(geo_scores > 0, as_tuple=False).flatten().detach().cpu().tolist()}
        for class_id in top_classes:
            score = float(mask_scores[class_id].detach().cpu())
            if uncertain_low <= score <= uncertain_high or int(class_id) in geo_class_ids:
                selected.append(int(class_id))
        return selected[: max(0, int(self.config.calibration.max_classes))]

    def _calibration_support_threshold(self) -> float:
        if self.config.calibration.pixel_support_threshold is not None:
            return float(self.config.calibration.pixel_support_threshold)
        return self._prob_threshold()

    def _limit_changed_pixels(self, raw_logits: torch.Tensor, calibrated_logits: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        max_changed_ratio = float(self.config.calibration.max_changed_ratio)
        if max_changed_ratio <= 0.0:
            return raw_logits.detach().clone(), 0.0, True

        raw_seg = self._logits_to_segmentation(raw_logits)
        calibrated_seg = self._logits_to_segmentation(calibrated_logits)
        changed = raw_seg != calibrated_seg
        changed_count = int(changed.sum().detach().cpu())
        total_count = int(changed.numel())
        changed_ratio = changed_count / max(1, total_count)
        if changed_count == 0 or changed_ratio <= max_changed_ratio:
            return calibrated_logits, float(changed_ratio), False

        keep_count = max(1, int(total_count * max_changed_ratio))
        raw_conf = raw_logits.max(dim=0)[0]
        calibrated_conf = calibrated_logits.max(dim=0)[0]
        gains = (calibrated_conf - raw_conf).detach().flatten()
        changed_flat = changed.flatten()
        changed_indices = torch.nonzero(changed_flat, as_tuple=False).flatten()
        changed_gains = gains[changed_indices]
        if changed_indices.numel() > keep_count:
            keep_local = torch.topk(changed_gains, k=keep_count).indices
            keep_indices = changed_indices[keep_local]
            keep_flat = torch.zeros_like(changed_flat, dtype=torch.bool)
            keep_flat[keep_indices] = True
            revert_flat = changed_flat & ~keep_flat
        else:
            revert_flat = torch.zeros_like(changed_flat, dtype=torch.bool)

        safe_logits = calibrated_logits.detach().clone()
        revert_mask = revert_flat.view_as(changed)
        if revert_mask.any():
            safe_logits[:, revert_mask] = raw_logits[:, revert_mask]
        safe_ratio = float((self._logits_to_segmentation(raw_logits) != self._logits_to_segmentation(safe_logits)).sum().detach().cpu()) / max(1, total_count)
        return safe_logits, safe_ratio, True

    def _build_exemplar_bank(self, raw: dict[str, torch.Tensor], raw_seg: torch.Tensor) -> list[dict[str, Any]]:
        threshold = self._mask_threshold()
        bank: list[dict[str, Any]] = []
        for class_id, class_name in enumerate(self.class_names[: raw["class_logits"].shape[0]]):
            class_mask = raw_seg == class_id
            if class_mask.sum() == 0:
                continue
            presence = float(raw["presence"][class_id].detach().cpu())
            mask_score = float(raw["mask_scores"][class_id].detach().cpu())
            head_agree = float(raw["head_agreement"][class_id].detach().cpu())
            if (
                presence < float(self.config.thresholds.exemplar_presence)
                or mask_score < float(self.config.thresholds.exemplar_mask_score)
                or head_agree < float(self.config.thresholds.exemplar_head_iou)
            ):
                continue
            bbox = mask_bbox(class_mask)
            if bbox is None:
                continue
            bank.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "presence_score": presence,
                    "mask_score": mask_score,
                    "head_agreement": head_agree,
                    "bbox": list(bbox),
                    "area": int(class_mask.sum().detach().cpu()),
                    "threshold": threshold,
                }
            )
        return bank

    def _save_artifacts(
        self,
        image_id: str,
        output_path: Path | None,
        raw: dict[str, torch.Tensor],
        raw_seg: torch.Tensor,
        calibrated_logits: torch.Tensor,
        calibrated_seg: torch.Tensor,
        save_enabled: bool,
    ) -> dict[str, dict[str, str]]:
        if output_path is None or not save_enabled:
            return {}

        raw_mask_path = output_path / f"{image_id}_tfcc_raw_mask.png"
        calibrated_mask_path = output_path / f"{image_id}_tfcc_calibrated_mask.png"
        Image.fromarray(raw_seg.detach().cpu().numpy().astype(np.uint8)).save(raw_mask_path)
        Image.fromarray(calibrated_seg.detach().cpu().numpy().astype(np.uint8)).save(calibrated_mask_path)

        paths = {
            "raw_result": {"final_mask_path": str(raw_mask_path)},
            "calibrated_result": {"final_mask_path": str(calibrated_mask_path)},
        }
        if self.config.output.save_dense_arrays:
            dense_path = output_path / f"{image_id}_tfcc_dense_outputs.npz"
            np.savez_compressed(
                dense_path,
                raw_logit=raw["class_logits"].detach().cpu().numpy().astype(np.float16),
                semantic_score_map=raw["semantic_raw_logits"].detach().cpu().numpy().astype(np.float16),
                instance_score_map=raw["instance_raw_logits"].detach().cpu().numpy().astype(np.float16),
                calibrated_logit=calibrated_logits.detach().cpu().numpy().astype(np.float16),
                raw_final_mask=raw_seg.detach().cpu().numpy().astype(np.uint8),
                calibrated_final_mask=calibrated_seg.detach().cpu().numpy().astype(np.uint8),
            )
            paths["raw_result"]["dense_output_path"] = str(dense_path)
            paths["calibrated_result"]["dense_output_path"] = str(dense_path)
        return paths

    def _json_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_id": result["image_id"],
            "debug": result.get("debug", {}),
            "exemplar_bank": result.get("exemplar_bank", []),
        }
        raw_result = result.get("raw_result")
        if raw_result is not None:
            payload["raw_result"] = {
                "mask_scores": raw_result.get("mask_scores", []),
                "presence": raw_result.get("presence", []),
                "head_agreement": raw_result.get("head_agreement", []),
                "artifact_paths": raw_result.get("artifact_paths", {}),
            }
        calibrated_result = result.get("calibrated_result")
        if calibrated_result is not None:
            payload["calibrated_result"] = {
                "geo_scores": calibrated_result.get("geo_scores", []),
                "artifact_paths": calibrated_result.get("artifact_paths", {}),
            }
        return _json_safe(payload)

    def _query_indices_for_classes(self, class_ids: Sequence[int] | None) -> list[int]:
        if class_ids is None:
            return list(range(int(getattr(self.base_model, "num_queries", len(self.base_model.query_words)))))
        query_idx = getattr(self.base_model, "query_idx")
        if torch.is_tensor(query_idx):
            query_idx_list = query_idx.detach().cpu().tolist()
        else:
            query_idx_list = list(query_idx)
        class_set = {int(item) for item in class_ids}
        return [index for index, class_id in enumerate(query_idx_list) if int(class_id) in class_set]

    def _logits_to_segmentation(self, logits: torch.Tensor) -> torch.Tensor:
        seg_pred = torch.argmax(logits, dim=0)
        max_vals = logits.max(dim=0)[0]
        seg_pred[max_vals < self._prob_threshold()] = self._bg_idx()
        return seg_pred

    def _mask_threshold(self) -> float:
        if self.config.thresholds.mask_threshold is not None:
            return float(self.config.thresholds.mask_threshold)
        return self._prob_threshold()

    def _prob_threshold(self) -> float:
        if self.config.prob_threshold is not None:
            return float(self.config.prob_threshold)
        return float(getattr(self.base_model, "prob_thd", 0.0))

    def _bg_idx(self) -> int:
        if self.config.bg_idx is not None:
            return int(self.config.bg_idx)
        return int(getattr(self.base_model, "bg_idx", 0))

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


def _tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().float().cpu().tolist()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "logits"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
