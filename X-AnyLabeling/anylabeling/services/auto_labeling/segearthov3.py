import logging
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.services.auto_labeling.model import Model
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.services.auto_labeling.utils.general import refine_contours
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

logger = logging.getLogger(__name__)


class SegEarthOV3(Model):
    """SegEarth-OV-3 open-vocabulary semantic segmentation for remote sensing."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "segearthov3_path",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        # Read config
        segearthov3_path = self.config.get(
            "segearthov3_path", ""
        )
        model_path = self.config.get("model_path", "")
        self.classes = self.config.get(
            "classes",
            ["background", "bareland", "grass", "road", "tree", "water", "cropland", "building", "car"],
        )
        self.prob_thd = self.config.get("prob_thd", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.1)
        self.slide_stride = self.config.get("slide_stride", 512)
        self.slide_crop = self.config.get("slide_crop", 512)
        self.bg_idx = self.config.get("bg_idx", 0)
        self.epsilon_factor = self.config.get("epsilon_factor", 0.001)

        # [ADDED] Category-Adaptive Dual-Head Fusion 파라미터 (Novelty Idea 1)
        # fusion_mode에 따라 두 헤드 결합 전략(max / heuristic / entropy)을 선택하고,
        # things/stuff 카테고리별로 서로 다른 α 가중치를 적용한다.
        self.fusion_mode  = self.config.get("fusion_mode", "max")
        self.things_alpha = self.config.get("things_alpha", 0.8)
        self.stuff_alpha  = self.config.get("stuff_alpha", 0.2)
        things_categories = self.config.get(
            "things_categories",
            ["building", "car"],
        )
        self.things_set = set(c.lower() for c in things_categories)

        # Validate paths
        if not os.path.isdir(segearthov3_path):
            raise FileNotFoundError(
                f"SegEarth-OV-3 project path not found: {segearthov3_path}"
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SAM3 weights not found: {model_path}"
            )

        # Add SegEarth-OV-3 to sys.path for imports
        if segearthov3_path not in sys.path:
            sys.path.insert(0, segearthov3_path)

        # Import SegEarth-OV-3 components
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build SAM3 model
        bpe_path = os.path.join(
            segearthov3_path, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
        on_message(
            self.tr("Loading SegEarth-OV-3 model... This may take a while.")
        )
        sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=model_path,
            device=str(self.device),
        )
        self.processor = Sam3Processor(
            sam3_model,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )

        # Parse class names (support comma-separated synonyms)
        self._parse_classes()
        self._build_alpha_vector()  # [ADDED] 카테고리별 α 벡터 사전 계산 (heuristic 모드용)

        on_message(
            self.tr("SegEarth-OV-3 model loaded successfully.")
        )

    def _parse_classes(self):
        """Parse class list into query words and indices."""
        self.query_words = []
        self.query_idx = []
        for idx, cls_entry in enumerate(self.classes):
            names = [n.strip() for n in cls_entry.split(",")]
            self.query_words.extend(names)
            self.query_idx.extend([idx] * len(names))

        self.num_cls = len(self.classes)
        self.num_queries = len(self.query_words)
        self.query_idx_tensor = torch.tensor(
            self.query_idx, dtype=torch.int64, device=self.device
        )

    # [NEW METHOD] heuristic 융합 모드에서 사용할 카테고리별 고정 α 가중치 벡터를
    # query_word 단위로 미리 계산해 GPU 텐서로 저장한다.
    # things 카테고리(건물·차량 등 개체)는 things_alpha, 나머지 stuff는 stuff_alpha 적용.
    def _build_alpha_vector(self):
        """카테고리별 α_c 벡터를 미리 계산한다. (query_word 단위)"""
        alphas = []
        for idx, word in enumerate(self.query_words):
            cls_idx  = self.query_idx[idx]
            cls_name = self.classes[cls_idx].split(",")[0].strip().lower()
            if cls_name in self.things_set:
                alphas.append(self.things_alpha)
            else:
                alphas.append(self.stuff_alpha)
        # shape: (num_queries,) — GPU에 올려두어 루프 내 인덱싱 가능
        self.alpha_vec = torch.tensor(
            alphas, dtype=torch.float32, device=self.device
        )

    # [NEW METHOD] entropy 융합 모드에서 사용할 픽셀별 동적 α 계산 메서드.
    # 두 헤드의 픽셀별 예측 엔트로피를 비교해 α_map (H×W)을 런타임에 결정한다.
    # instance 헤드가 확실할수록(H_inst↓, confidence↑) α↑ → instance 헤드에 더 많은 가중치 부여.
    @staticmethod
    def _compute_entropy_alpha(
        inst_logits: torch.Tensor,
        sem_logits: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        두 헤드의 픽셀별 예측 엔트로피로 α_map (H×W)을 동적 계산한다.

        Args:
            inst_logits: P_inst_agg  (H, W) — sigmoid 이전 logit
            sem_logits:  P_sem       (H, W) — sigmoid 이전 logit
        Returns:
            alpha_map: Tensor (H, W), values in [0, 1]
                       높을수록 instance head에 더 많은 가중치
        """
        import math

        def _binary_entropy(logits: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(logits.float())            # bfloat16 대비 float32
            p = p.clamp(eps, 1.0 - eps)
            # log2 정규화 → 엔트로피 범위 [0, 1]
            return -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)

        h_inst = _binary_entropy(inst_logits)            # (H, W), range [0, 1]
        h_sem  = _binary_entropy(sem_logits)             # (H, W), range [0, 1]

        conf_inst = 1.0 - h_inst                         # confidence = 1 - entropy
        conf_sem  = 1.0 - h_sem
        total     = (conf_inst + conf_sem).clamp(min=eps)
        return conf_inst / total                         # α_map: inst 확실할수록 α↑

    def _inference_single_view(self, image):
        """Run inference on a single PIL image."""
        w, h = image.size
        seg_logits = torch.zeros(
            (self.num_queries, h, w), device=self.device
        )

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state, prompt=query_word
                )


                # Novelty Idea 1: Category-Adaptive Dual-Head Fusion

                # ── [HEAD 1] Instance Head ─────────────────────────────────────
                # N개의 instance mask logit을 object_score와 곱한 뒤 element-wise MAX로 집계
                # 결과: P_inst_agg (shape: H×W)
                if inference_state["masks_logits"].shape[0] > 0:
                    inst_len = inference_state["masks_logits"].shape[0]
                    for inst_id in range(inst_len):
                        instance_logits = inference_state["masks_logits"][
                            inst_id
                        ].squeeze()
                        instance_score = inference_state["object_score"][
                            inst_id
                        ]
                        # (resize if needed)
                        if instance_logits.shape != (h, w):
                            instance_logits = F.interpolate(
                                instance_logits.view(
                                    1, 1, *instance_logits.shape
                                ),
                                size=(h, w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        seg_logits[query_idx] = torch.max(
                            seg_logits[query_idx],
                            instance_logits * instance_score,       # P_inst_i × score_i
                        )


                # ── [HEAD 2] Semantic Head ─────────────────────────────────────
                # → seg_logits[query_idx] = P_inst_agg = MAX_i(P_inst_i × score_i)
                # 전역 semantic mask logit을 instance 집계 결과와 element-wise MAX로 융합
                # 결과: P_fused = max(P_inst_agg, P_sem)  ← 논문 Eq. 2
                semantic_logits = inference_state["semantic_mask_logits"]
                if semantic_logits.shape != (h, w):
                    semantic_logits = F.interpolate(
                        semantic_logits
                        if semantic_logits.dim() == 4
                        else semantic_logits.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()

                # ── Dual-Head Fusion ─────────────────────────────────────────────
                # [CHANGED] 기존 단순 max 고정 융합 → fusion_mode에 따라 세 가지 전략 분기:
                #   max      : element-wise max (원래 방식 유지)
                #   heuristic: 카테고리별 고정 α로 가중 평균 (things vs stuff 차등)
                #   entropy  : 두 헤드의 예측 엔트로피로 α를 런타임에 동적 결정
                inst_logits = seg_logits[query_idx].clone()   # P_inst_agg

                if self.fusion_mode == "max":
                    seg_logits[query_idx] = torch.max(
                        inst_logits, semantic_logits
                    )

                elif self.fusion_mode == "heuristic":
                    alpha = self.alpha_vec[query_idx]
                    seg_logits[query_idx] = (
                        alpha * inst_logits.float()
                        + (1.0 - alpha) * semantic_logits.float()
                    )

                elif self.fusion_mode == "entropy":
                    alpha_map = self._compute_entropy_alpha(
                        inst_logits, semantic_logits
                    )
                    seg_logits[query_idx] = (
                        alpha_map * inst_logits.float()
                        + (1.0 - alpha_map) * semantic_logits.float()
                    )


                # ── [Presence Score] ──────────────────────────────────────────
                # 클래스 존재 확률로 스케일링 → false positive 억제
                seg_logits[query_idx] = (
                    seg_logits[query_idx] * inference_state["presence_score"]
                )


        return seg_logits

    def _slide_inference(self, image):
        """Sliding window inference for large images."""
        w_img, h_img = image.size
        stride = (self.slide_stride, self.slide_stride)
        crop_size = (self.slide_crop, self.slide_crop)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros(
            (self.num_queries, h_img, w_img), device=self.device
        )
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)

                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        preds = preds / count_mat
        return preds

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """Run SegEarth-OV-3 inference and convert results to shapes."""
        if image is None and filename is None:
            return AutoLabelingResult([], replace=False)

        try:
            # Convert QImage to numpy RGB
            cv_image = qt_img_to_rgb_cv_img(image, filename)

            # Convert numpy RGB to PIL Image
            pil_image = PILImage.fromarray(cv_image)
            h, w = cv_image.shape[:2]

            # Choose inference mode
            if self.slide_crop > 0 and (
                self.slide_crop < w or self.slide_crop < h
            ):
                seg_logits = self._slide_inference(pil_image)
            else:
                seg_logits = self._inference_single_view(pil_image)

            # Aggregate class synonyms if needed
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = F.one_hot(
                    self.query_idx_tensor, num_classes=self.num_cls
                )
                cls_index = cls_index.T.view(
                    self.num_cls, self.num_queries, 1, 1
                ).float()
                seg_logits = (seg_logits * cls_index).max(1)[0]

            # Get predictions
            seg_pred = torch.argmax(seg_logits, dim=0)
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            seg_pred_np = seg_pred.cpu().numpy()

            # Convert segmentation mask to polygon shapes
            shapes = self._mask_to_shapes(seg_pred_np, h, w)

            return AutoLabelingResult(shapes, replace=True)

        except Exception as e:
            logger.error(f"SegEarth-OV-3 inference error: {e}")
            return AutoLabelingResult([], replace=False)

    def _mask_to_shapes(self, seg_pred, h, w):
        """Convert segmentation mask to Shape objects."""
        shapes = []
        img_area = h * w

        for class_idx in range(self.num_cls):
            if class_idx == self.bg_idx:
                continue

            binary_mask = (seg_pred == class_idx).astype(np.uint8) * 255
            if binary_mask.sum() == 0:
                continue

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue

            contours = refine_contours(
                contours, img_area, epsilon_factor=self.epsilon_factor
            )

            class_name = self.classes[class_idx]
            # Use the first name if comma-separated synonyms
            label = class_name.split(",")[0].strip()

            for contour in contours:
                points = contour.reshape(-1, 2).tolist()
                if len(points) < 3:
                    continue

                shape = Shape(flags={})
                for pt in points:
                    shape.add_point(QtCore.QPointF(pt[0], pt[1]))
                shape.shape_type = (
                    "polygon"
                    if self.output_mode == "polygon"
                    else "rectangle"
                )
                shape.closed = True
                shape.label = label

                if self.output_mode == "rectangle":
                    # Convert polygon to bounding box
                    x_coords = [pt[0] for pt in points]
                    y_coords = [pt[1] for pt in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    shape = Shape(flags={})
                    shape.add_point(QtCore.QPointF(x_min, y_min))
                    shape.add_point(QtCore.QPointF(x_max, y_min))
                    shape.add_point(QtCore.QPointF(x_max, y_max))
                    shape.add_point(QtCore.QPointF(x_min, y_max))
                    shape.shape_type = "rectangle"
                    shape.closed = True
                    shape.label = label

                shapes.append(shape)

        return shapes

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_conf(self, value):
        """Update probability threshold from UI slider."""
        self.prob_thd = value

    def unload(self):
        """Release GPU memory."""
        if hasattr(self, "processor"):
            del self.processor
        torch.cuda.empty_cache()
