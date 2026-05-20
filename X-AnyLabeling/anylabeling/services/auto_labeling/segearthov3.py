import logging
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
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


class _PromptInjectingProcessor:
    """Sam3Processor 를 감싸 set_text_prompt 호출 직후 큐잉된 spatial prompt
    (box) 를 자동으로 add_geometric_prompt 로 주입한다.

    그 외 모든 메서드(set_image, reset_all_prompts, _forward_grounding 등)는
    __getattr__ 로 inner processor 에 그대로 위임된다.
    """

    def __init__(self, inner, wrapper):
        # __setattr__ 가 inner 로 위임되지 않도록 직접 __dict__ 에 기록
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_wrapper", wrapper)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_text_prompt(self, *args, **kwargs):
        # SAM3 의 native signature: set_text_prompt(prompt: str, state: Dict)
        # X-AnyLabeling baseline / RCR evidence_extractor 모두 keyword 호출
        # (state=..., prompt=...) 패턴을 쓰므로 양쪽 모두 수용한다.
        if "prompt" in kwargs:
            prompt = kwargs.pop("prompt")
        elif args:
            prompt = args[0]
            args = args[1:]
        else:
            raise TypeError("set_text_prompt requires 'prompt'")
        if "state" in kwargs:
            state = kwargs.pop("state")
        elif args:
            state = args[0]
            args = args[1:]
        else:
            raise TypeError("set_text_prompt requires 'state'")
        if kwargs or args:
            raise TypeError(f"Unexpected args for set_text_prompt: {args} {kwargs}")

        state = self._inner.set_text_prompt(prompt, state)

        wrapper = self._wrapper
        pending = getattr(wrapper, "_pending_spatial_prompts", None)
        if not pending:
            return state
        class_id = wrapper._query_word_to_class_id(prompt)
        if class_id is None:
            return state
        for entry in pending.get(class_id, []):
            try:
                state = self._inner.add_geometric_prompt(
                    entry["box_xywh_norm"], entry["label"], state
                )
            except Exception as exc:  # spatial prompt 실패는 inference 전체를 막지 않음
                logger.warning(
                    f"add_geometric_prompt failed for class_id={class_id}: {exc}"
                )
        return state


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
            # ── [ADDED] Exemplar / Spatial prompt UI ──────────────────────────
            "button_exemplar_mode",
            "edit_exemplar_class",
            "exemplar_shape_combo",
            "exemplar_radius_spinbox",
            "button_exemplar_add_class",
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

        # ── [ADDED] Multi-scale Adapter 설정 파싱 ──────────────────────────────
        # adapter_mode: "off"  → RSAdapter 비활성화 (기존 동작 유지)
        #               "hook" → Forward Hook 기반 추론 전용 모드
        #               "full" → 학습된 어댑터 가중치 로드 모드
        self.adapter_mode       = self.config.get("adapter_mode", "off")
        self.adapter_bottleneck = self.config.get("adapter_bottleneck", 64)
        self.adapter_path       = self.config.get("adapter_path", "")

        # ── [ADDED] RCR-SegEarth 설정 파싱 ─────────────────────────────────────
        # use_rcr=True 인 경우 baseline 추론 결과 대신 RCRInferencer 가 반환한
        # refined segmentation 을 사용한다.
        self.use_rcr          = bool(self.config.get("use_rcr", False))
        self.rcr_config_path  = self.config.get("rcr_config_path", "") or ""
        self.rcr_output_dir   = self.config.get("rcr_output_dir", "") or ""
        self.rcr_save_json    = bool(self.config.get("rcr_save_json", False))
        # RCREvidenceExtractor 가 getattr(base_model, "use_presence_score", True)
        # 로 참조 → 명시적으로 노출하여 baseline 과 일관된 동작을 보장한다.
        self.use_presence_score = bool(self.config.get("use_presence_score", True))
        self._rcr_inferencer  = None
        self._sam3_model      = None  # parameters() 셰임용 핸들

        # ── [ADDED] Exemplar (spatial prompt + novel class) 상태 ──────────────
        # _pending_spatial_prompts: 다음 predict_shapes 호출 1회 동안만 적용되는
        # spatial box prompt 큐. 키는 class_id(int), 값은 entry dict 리스트.
        # entry 스키마: {"box_xywh_norm": [cx,cy,w,h] in [0,1], "label": bool}
        self._last_cv_image   = None
        self._last_filename   = None
        self._pending_spatial_prompts = {}
        # Sam3Processor 의 set_text_prompt 호출 시 prompt 문자열 → class_id 역매핑
        # 에 사용 (대소문자 무시, _parse_classes 마다 재구성).
        self._query_word_lookup = {}

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
        inner_processor = Sam3Processor(
            sam3_model,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        # [ADDED] Sam3Processor 를 proxy 로 감싸 set_text_prompt 직후 pending
        # spatial box prompt 를 자동 주입한다. proxy 는 __getattr__ 로 나머지
        # 메서드를 그대로 위임하므로 기존 호출부(baseline + RCR) 모두 수정 없이
        # 동일 인터페이스로 동작한다.
        self.processor = _PromptInjectingProcessor(inner_processor, self)
        # RCRInferencer 가 base_model.parameters() 로 requires_grad_=False 설정
        # 을 시도하므로 SAM3 모델 핸들을 보관해 두고 parameters() 셰임에 사용한다.
        self._sam3_model = sam3_model

        # Parse class names (support comma-separated synonyms)
        self._parse_classes()
        self._build_alpha_vector()  # [ADDED] 카테고리별 α 벡터 사전 계산 (heuristic 모드용)

        # ── [ADDED] Multi-scale Adapter 모듈 초기화 + Forward Hook 등록 ─────────
        # sam3_model, segearthov3_path, self.device 모두 이 시점에 유효하다.
        if self.adapter_mode in ("hook", "full"):
            adapter_dir = os.path.join(segearthov3_path, "rs_adapter")
            if adapter_dir not in sys.path:
                sys.path.insert(0, adapter_dir)
            from rs_adapter import RSAdapter, RSMultiscaleFPN

            # ViT 블록 리스트 접근
            # sam3_model.backbone           → SAM3VLBackbone (vl_combiner.py)
            # .vision_backbone              → Sam3DualViTDetNeck (necks.py)
            # .trunk                        → ViT (vitdet.py)
            # .blocks                       → nn.ModuleList (len=32)
            vit_trunk = sam3_model.backbone.vision_backbone.trunk
            self._vit_blocks = vit_trunk.blocks  # nn.ModuleList, len=32

            # RSAdapter 초기화 (블록 수만큼)
            self._adapters = nn.ModuleList([
                RSAdapter(d_model=1024, bottleneck=self.adapter_bottleneck)
                for _ in range(len(self._vit_blocks))
            ]).to(self.device)

            # FPN 모듈 초기화
            self._fpn = RSMultiscaleFPN(in_channels=1024, out_channels=256).to(self.device)

            # 학습된 가중치 로드 (full 모드, 경로가 지정된 경우)
            if self.adapter_path and os.path.isfile(self.adapter_path):
                ckpt = torch.load(self.adapter_path, map_location=self.device)
                self._adapters.load_state_dict(ckpt.get("adapters", ckpt))
                self._fpn.load_state_dict(ckpt.get("fpn", {}), strict=False)
                logger.info(f"RSAdapter 가중치 로드 완료: {self.adapter_path}")

            # SAM3 파라미터 동결 (어댑터 + FPN만 학습 가능)
            for param in sam3_model.parameters():
                param.requires_grad = False
            for param in self._adapters.parameters():
                param.requires_grad = True
            for param in self._fpn.parameters():
                param.requires_grad = True

            # ── Forward Hook 등록 ────────────────────────────────────────────
            # 체크포인트 블록(7, 15, 23, 31): 출력 포착 + 어댑터 적용
            # 나머지 블록: 어댑터 적용만
            # PyTorch hook이 None이 아닌 값을 반환하면 해당 값이 블록 출력을 대체한다.
            self._hook_feats = {}
            self._hooks = []
            checkpoint_blocks = {7: "f7", 15: "f15", 23: "f23", 31: "f31"}

            def _make_hook(adapter_module, feat_key):
                def _hook(module, input, output):
                    adapted = adapter_module(output)
                    self._hook_feats[feat_key] = adapted.detach()
                    return adapted
                return _hook

            for blk_idx, feat_key in checkpoint_blocks.items():
                hook = self._vit_blocks[blk_idx].register_forward_hook(
                    _make_hook(self._adapters[blk_idx], feat_key)
                )
                self._hooks.append(hook)

            non_checkpoint = set(range(len(self._vit_blocks))) - set(checkpoint_blocks.keys())
            for blk_idx in non_checkpoint:
                adapter = self._adapters[blk_idx]
                hook = self._vit_blocks[blk_idx].register_forward_hook(
                    lambda m, i, o, a=adapter: a(o)
                )
                self._hooks.append(hook)

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
        # [ADDED] proxy 의 set_text_prompt hook 에서 사용할 역매핑.
        # prompt 문자열(소문자) → 그 prompt 가 속한 클래스 id.
        self._query_word_lookup = {
            self.query_words[i].strip().lower(): self.query_idx[i]
            for i in range(self.num_queries)
        }

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
            # set_image() 호출 전 hook 수집 버퍼를 초기화한다.
            # (슬라이딩 윈도우 모드에서 패치마다 새 피처를 수집하기 위해 필수)
            if self.adapter_mode in ("hook", "full"):
                self._hook_feats.clear()
            inference_state = self.processor.set_image(image)

            # ── [ADDED] Multi-scale Adapter FPN 퓨전 ─────────────────────────
            # set_image() 내부 ViT forward 중 hook이 실행 → _hook_feats에 중간 피처 수집
            # 4개 체크포인트 블록(7,15,23,31) 피처가 모두 수집된 경우에만 FPN 적용
            if self.adapter_mode in ("hook", "full") and len(self._hook_feats) == 4:
                fpn_out = self._fpn(self._hook_feats)
                # P4 (72×72, 256ch)를 기존 vision_features 자리에 주입
                # → Transformer Encoder가 이 피처를 크로스 어텐션에 사용
                inference_state["backbone_out"]["vision_features"] = fpn_out["p4"]
                # backbone_fpn 전체도 교체 (멀티레벨 어텐션 확장 대비)
                inference_state["backbone_out"]["backbone_fpn"] = [
                    fpn_out["p2"],  # (B, 256, 288, 288)
                    fpn_out["p3"],  # (B, 256, 144, 144)
                    fpn_out["p4"],  # (B, 256, 72,  72)
                    fpn_out["p5"],  # (B, 256, 36,  36)
                ]

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

            # [ADDED] 이미지 캐시 및 ephemeral spatial-prompt 큐 라이프사이클 관리.
            # filename 이 바뀌면 직전 이미지에서 큐잉됐던 좌표가 새 이미지에
            # 잘못 적용되는 것을 막기 위해 자동 폐기한다.
            if filename and filename != self._last_filename:
                if self._pending_spatial_prompts:
                    logger.info(
                        "cleared pending spatial prompts (filename change "
                        f"{self._last_filename!r} -> {filename!r})"
                    )
                self._pending_spatial_prompts.clear()
            self._last_cv_image = cv_image
            self._last_filename = filename

            # ── [ADDED] RCR-SegEarth 분기 ─────────────────────────────────────
            # use_rcr=True 인 경우 baseline 추론을 우회하고 RCRInferencer 가
            # 반환한 refined segmentation_map 을 그대로 사용한다.
            if self.use_rcr:
                seg_pred_np = self._run_rcr_inference(pil_image, filename)
            else:
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
                # baseline 경로도 ephemeral 큐를 소비한 것으로 간주.
                self._pending_spatial_prompts.clear()

            # Convert segmentation mask to polygon shapes
            shapes = self._mask_to_shapes(seg_pred_np, h, w)

            return AutoLabelingResult(shapes, replace=True)

        except Exception as e:
            logger.error(f"SegEarth-OV-3 inference error: {e}")
            return AutoLabelingResult([], replace=False)

    # ── [ADDED] RCR-SegEarth 통합 헬퍼 ────────────────────────────────────────
    def parameters(self):
        """RCRInferencer 가 base_model.parameters() 로 grad 동결을 시도하므로
        SAM3 모델의 파라미터 이터레이터를 위임 노출한다.
        """
        if self._sam3_model is None:
            return iter([])
        return self._sam3_model.parameters()

    def _get_rcr_inferencer(self):
        """RCRInferencer 를 lazy 초기화한다. segearthov3_path 가 이미
        sys.path 에 추가돼 있으므로 rcr 패키지를 바로 import 할 수 있다."""
        if self._rcr_inferencer is not None:
            return self._rcr_inferencer
        from rcr.rcr_inferencer import RCRInferencer

        rcr_config = self.rcr_config_path if self.rcr_config_path else None
        if rcr_config and not os.path.isfile(rcr_config):
            logger.warning(
                f"RCR config not found at {rcr_config}, falling back to defaults."
            )
            rcr_config = None

        self._rcr_inferencer = RCRInferencer(
            base_model=self,
            class_names=list(self.classes),
            config=rcr_config,
            output_dir=(self.rcr_output_dir or None),
            device=self.device,
        )
        return self._rcr_inferencer

    def _run_rcr_inference(self, pil_image, filename):
        """RCRInferencer.infer_image 를 호출하고 (H, W) 정수 클래스 맵을 반환한다.

        spatial prompt 큐가 비어있지 않으면 RCR 의 TTA(consensus) 단계만 일시
        비활성한다. boundary / local_vote / component / safety 등 나머지 RCR
        단계는 그대로 적용. inference 직후 ephemeral 큐를 비운다.
        """
        inferencer = self._get_rcr_inferencer()
        has_spatial = bool(self._pending_spatial_prompts)
        prev_tta_enabled = inferencer.config.tta.enabled
        if has_spatial:
            inferencer.config.tta.enabled = False
            logger.info(
                f"RCR TTA disabled (spatial active for class_ids="
                f"{sorted(self._pending_spatial_prompts.keys())})"
            )

        image_id = None
        if filename:
            image_id = os.path.splitext(os.path.basename(str(filename)))[0]
        try:
            result = inferencer.infer_image(
                pil_image,
                image_id=image_id,
                output_dir=(self.rcr_output_dir or None),
                save_json=self.rcr_save_json,
            )
        finally:
            inferencer.config.tta.enabled = prev_tta_enabled
            # ephemeral: 1회 소비. 동일 이미지 재실행 시 새 spatial 입력 필요.
            self._pending_spatial_prompts.clear()

        seg_map = result.get("segmentation_map")
        if seg_map is None:
            # 폴백: logits 가 있다면 argmax + prob_thd 로 mask 생성
            logits = result.get("logits")
            if torch.is_tensor(logits):
                seg_pred = torch.argmax(logits, dim=0)
                max_vals = logits.max(0)[0]
                seg_pred[max_vals < self.prob_thd] = self.bg_idx
                return seg_pred.detach().cpu().numpy().astype(np.int32)
            raise RuntimeError("RCRInferencer returned no segmentation output.")
        return np.asarray(seg_map, dtype=np.int32)

    # ── [ADDED] Exemplar / Spatial prompt 통합 헬퍼 ──────────────────────────
    def _query_word_to_class_id(self, prompt):
        """proxy processor 가 set_text_prompt 호출 시 호출하는 역매핑.
        대소문자 무시. 매칭 실패 시 None.
        """
        if not isinstance(prompt, str):
            return None
        return self._query_word_lookup.get(prompt.strip().lower())

    def _resolve_or_register_class_id(self, label):
        """라벨 문자열을 기존 self.classes 와 매칭. 미존재 시 새 클래스로 등록.

        매칭은 case-insensitive, 클래스 엔트리의 동의어(comma-separated)도 검사.
        새 클래스가 등록되면 _parse_classes / _build_alpha_vector 재호출 +
        기 lazy-init 된 RCR inferencer 무효화 (class_names 캐시 갱신 위해).
        """
        if not isinstance(label, str):
            return None
        canon = label.strip()
        if not canon:
            return None
        canon_lower = canon.lower()
        for idx, cls_entry in enumerate(self.classes):
            for synonym in cls_entry.split(","):
                if synonym.strip().lower() == canon_lower:
                    return idx
        # 미존재 → 등록
        self.classes.append(canon)
        new_id = len(self.classes) - 1
        self._parse_classes()
        self._build_alpha_vector()
        # RCR inferencer 는 __init__ 에서 class_names 를 캐시하므로 무효화.
        # 다음 _get_rcr_inferencer 호출 시 새 class_names 로 재생성된다.
        self._rcr_inferencer = None
        logger.info(
            f"registered new class '{canon}' as id={new_id} "
            f"(num_cls={self.num_cls})"
        )
        return new_id

    @staticmethod
    def _clamp_xyxy(x1, y1, x2, y2, h, w):
        """이미지 경계 안으로 clamp + min/max 정렬."""
        x_min, x_max = sorted((float(x1), float(x2)))
        y_min, y_max = sorted((float(y1), float(y2)))
        x_min = max(0.0, min(x_min, float(w - 1)))
        x_max = max(0.0, min(x_max, float(w - 1)))
        y_min = max(0.0, min(y_min, float(h - 1)))
        y_max = max(0.0, min(y_max, float(h - 1)))
        return x_min, y_min, x_max, y_max

    def _xyxy_to_cxcywh_norm(self, x1, y1, x2, y2, h, w):
        """SAM3 add_geometric_prompt 가 요구하는 [cx,cy,w,h] in [0,1] 로 변환."""
        x_min, y_min, x_max, y_max = self._clamp_xyxy(x1, y1, x2, y2, h, w)
        bw = max(1.0, x_max - x_min)
        bh = max(1.0, y_max - y_min)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        return [cx / w, cy / h, bw / w, bh / h]

    def _point_to_cxcywh_norm(self, x, y, radius, h, w):
        """point + radius 를 동일 box 포맷으로 변환 (SAM3 public API 가 box 만)."""
        x = max(0.0, min(float(x), float(w - 1)))
        y = max(0.0, min(float(y), float(h - 1)))
        r = max(1.0, float(radius))
        bw = min(2.0 * r, float(w))
        bh = min(2.0 * r, float(h))
        return [x / w, y / h, bw / w, bh / h]

    def set_auto_labeling_marks(self, marks):
        """X-AnyLabeling auto_labeling widget 으로부터 exemplar 마크를 받아
        spatial prompt 큐 갱신 및 novel class 등록을 수행한다.

        marks 스키마(예시):
          {"type":"exemplar", "shape_type":"rectangle", "data":[x1,y1,x2,y2],
           "label":"<class name>"}
          {"type":"exemplar", "shape_type":"point", "data":[x,y],
           "radius": <px>, "label":"<class name>"}
          {"type":"exemplar", "shape_type":"text_only", "data":None,
           "label":"<class name>"}
        그 외 type 은 무시 (SAM 스타일 marks 등은 본 모델에 의미 없음).
        """
        if not isinstance(marks, list) or len(marks) == 0:
            return
        # text-only 마크는 이미지 캐시 없이도 가능하므로 spatial 마크 유무를
        # 먼저 확인한다.
        spatial_marks_present = any(
            isinstance(m, dict)
            and m.get("type") == "exemplar"
            and m.get("shape_type") in ("rectangle", "point")
            for m in marks
        )
        if spatial_marks_present and self._last_cv_image is None:
            logger.warning(
                "set_auto_labeling_marks: no cached image yet. "
                "Run inference once before injecting spatial exemplars."
            )
            return
        h, w = (
            self._last_cv_image.shape[:2]
            if self._last_cv_image is not None
            else (1, 1)
        )
        added = 0
        for mark in marks:
            if not isinstance(mark, dict) or mark.get("type") != "exemplar":
                continue
            label = mark.get("label", "")
            class_id = self._resolve_or_register_class_id(label)
            if class_id is None:
                logger.warning(
                    f"set_auto_labeling_marks: skipped mark with invalid label "
                    f"{label!r}"
                )
                continue
            shape_type = mark.get("shape_type", "")
            data = mark.get("data")
            if shape_type == "text_only":
                # 등록만 수행 (spatial 큐에 push 하지 않음)
                continue
            try:
                if shape_type == "rectangle":
                    if not (isinstance(data, (list, tuple)) and len(data) == 4):
                        raise ValueError(f"rectangle data shape: {data!r}")
                    box = self._xyxy_to_cxcywh_norm(
                        data[0], data[1], data[2], data[3], h, w
                    )
                elif shape_type == "point":
                    if not (isinstance(data, (list, tuple)) and len(data) == 2):
                        raise ValueError(f"point data shape: {data!r}")
                    radius = mark.get("radius", 12)
                    box = self._point_to_cxcywh_norm(
                        data[0], data[1], radius, h, w
                    )
                else:
                    logger.warning(
                        f"set_auto_labeling_marks: unsupported shape_type "
                        f"{shape_type!r}; skipped"
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    f"set_auto_labeling_marks: malformed mark {mark!r} ({exc})"
                )
                continue
            entry = {"box_xywh_norm": box, "label": True}
            self._pending_spatial_prompts.setdefault(class_id, []).append(entry)
            added += 1
        if added:
            logger.info(
                f"set_auto_labeling_marks: queued {added} spatial prompt(s); "
                f"pending={{cid: count}} = "
                f"{ {k: len(v) for k, v in self._pending_spatial_prompts.items()} }"
            )

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
        # [ADDED] forward hook 해제 (메모리 누수 방지)
        if hasattr(self, "_hooks"):
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

        if hasattr(self, "_rcr_inferencer"):
            self._rcr_inferencer = None
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "_adapters"):
            del self._adapters
        if hasattr(self, "_fpn"):
            del self._fpn
        self._sam3_model = None
        # [ADDED] exemplar 상태 정리
        self._pending_spatial_prompts = {}
        self._last_cv_image = None
        self._last_filename = None
        self._query_word_lookup = {}
        torch.cuda.empty_cache()
