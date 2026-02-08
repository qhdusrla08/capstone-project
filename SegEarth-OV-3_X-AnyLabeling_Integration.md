# SegEarth-OV-3 x X-AnyLabeling 통합 가이드

## 1. 프로젝트 개요

| 프로젝트 | 역할 | 경로 |
|----------|------|------|
| **SegEarth-OV-3** | Open-Vocabulary 시맨틱 세그멘테이션 (원격탐사) | `/home/yeon030108/capstone/SegEarth-OV-3/` |
| **X-AnyLabeling** | AI 기반 이미지 라벨링 도구 (PyQt5 GUI) | `/home/yeon030108/capstone/X-AnyLabeling/` |

**통합 목적**: X-AnyLabeling GUI에서 SegEarth-OV-3 모델을 선택하여 원격탐사 이미지를 자동 세그멘테이션하고, 결과를 폴리곤 어노테이션으로 변환한다.

---

## 2. 아키텍처

```
X-AnyLabeling GUI
    │
    ├─ 모델 선택: "SegEarth-OV-3 (SAM3)"
    │
    ├─ model_manager.py (_load_model)
    │       │
    │       └─ segearthov3.py (래퍼 클래스)
    │               │
    │               ├─ SAM3 모델 로드 (sam3.pt, 3.3GB)
    │               ├─ Sam3Processor.set_image() → 이미지 인코딩
    │               ├─ 클래스별 set_text_prompt() → 듀얼 헤드 추론
    │               ├─ seg_logits → argmax → 세그멘테이션 마스크
    │               └─ cv2.findContours() → Shape(polygon) 변환
    │
    └─ Canvas에 폴리곤 어노테이션 표시
```

---

## 3. 생성 및 수정한 파일

### 3.1 [NEW] `segearthov3.py` - 래퍼 모델 클래스

**경로**: `X-AnyLabeling/anylabeling/services/auto_labeling/segearthov3.py`

SegEarth-OV-3을 X-AnyLabeling의 `Model` 인터페이스에 맞게 감싸는 클래스.

```python
class SegEarthOV3(Model):
    class Meta:
        required_config_names = ["type", "name", "display_name", "model_path", "segearthov3_path"]
        widgets = ["button_run", "input_conf", "edit_conf", "toggle_preserve_existing_annotations"]
        output_modes = {"polygon": "Polygon", "rectangle": "Rectangle"}
        default_output_mode = "polygon"
```

**주요 메서드**:

| 메서드 | 역할 |
|--------|------|
| `__init__()` | SAM3 모델 빌드, Sam3Processor 초기화, 클래스명 파싱 |
| `predict_shapes(image, filename)` | QImage→numpy→PIL 변환 → 추론 → 마스크→폴리곤 변환 |
| `_inference_single_view(image)` | 단일 이미지 추론 (인스턴스+시맨틱 듀얼 헤드 융합) |
| `_slide_inference(image)` | 대형 이미지용 슬라이딩 윈도우 추론 |
| `_mask_to_shapes(seg_pred, h, w)` | 세그멘테이션 마스크 → Shape 폴리곤 객체 리스트 변환 |
| `set_auto_labeling_conf(value)` | UI 슬라이더에서 prob_thd 동적 변경 |
| `unload()` | GPU 메모리 해제 |

**마스크→폴리곤 변환 핵심 로직**:

```python
for class_idx in range(num_cls):
    binary_mask = (seg_pred == class_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = refine_contours(contours, img_area, epsilon_factor=0.001)
    for contour in contours:
        shape = Shape(shape_type="polygon", label=class_name, closed=True)
        for pt in contour.reshape(-1, 2):
            shape.add_point(QtCore.QPointF(pt[0], pt[1]))
```

### 3.2 [NEW] `segearthov3.yaml` - 모델 설정

**경로**: `X-AnyLabeling/anylabeling/configs/auto_labeling/segearthov3.yaml`

```yaml
type: segearthov3
name: segearthov3-sam3
display_name: "SegEarth-OV-3 (SAM3)"
model_path: /home/yeon030108/capstone/SegEarth-OV-3/weights/sam3/sam3.pt
segearthov3_path: /home/yeon030108/capstone/SegEarth-OV-3
classes:
  - background
  - "bareland,barren"
  - grass
  - road
  - "tree,forest"
  - "water,river"
  - cropland
  - "building,roof,house"
prob_thd: 0.1
confidence_threshold: 0.1
slide_stride: 512
slide_crop: 512
bg_idx: 0
epsilon_factor: 0.001
```

- `classes`: 쉼표로 동의어(synonym) 지정 가능 → 동일 클래스로 집계됨
- `slide_stride/slide_crop`: 대형 이미지를 512x512 패치로 분할 추론
- `prob_thd`: 확률 임계값 이하 픽셀은 background로 처리

### 3.3 [EDIT] `__init__.py` - 모델 타입 등록

**경로**: `X-AnyLabeling/anylabeling/services/auto_labeling/__init__.py`

3개 리스트에 `"segearthov3"` 추가:

```python
_CUSTOM_MODELS = [..., "segearthov3"]                                    # 모델 타입 인식
_AUTO_LABELING_CONF_MODELS = [..., "segearthov3"]                        # confidence 슬라이더 활성화
_AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS = [..., "segearthov3"]  # 기존 어노테이션 보존 옵션
```

### 3.4 [EDIT] `model_manager.py` - 모델 로딩 분기 추가

**경로**: `X-AnyLabeling/anylabeling/services/auto_labeling/model_manager.py`

`_load_model()` 메서드의 `else: raise` 직전에 elif 블록 추가:

```python
elif model_config["type"] == "segearthov3":
    from .segearthov3 import SegEarthOV3
    try:
        model_config["model"] = SegEarthOV3(
            model_config, on_message=self.new_model_status.emit
        )
        self.auto_segmentation_model_unselected.emit()
        logger.info(f"Model loaded successfully: {model_config['type']}")
    except Exception as e:
        # 에러 처리 (기존 패턴과 동일)
```

### 3.5 [EDIT] `models.yaml` - 모델 레지스트리 등록

**경로**: `X-AnyLabeling/anylabeling/configs/models.yaml`

```yaml
- model_name: "segearthov3-sam3"
  config_file: ":/segearthov3.yaml"
```

---

## 4. 환경 설정

### 4.1 가상환경

**사용 환경**: `segearth_stable` (conda)

```bash
conda activate segearth_stable
```

### 4.2 추가 설치한 패키지

`segearth_stable` 환경에 X-AnyLabeling 실행을 위해 추가 설치한 패키지:

```bash
pip install natsort PyQt5 PyQtWebEngine qimage2ndarray lapx \
    importlib_metadata json_repair jsonlines openai pyclipper shapely tokenizers
```

### 4.3 Qt 플러그인 충돌 해결

`opencv-python`에 번들된 Qt와 `PyQt5`의 Qt가 충돌하므로 headless 버전으로 교체:

```bash
pip install opencv-contrib-python-headless==4.9.0.80
pip uninstall opencv-python -y
```

### 4.4 최종 핵심 패키지 버전

| 패키지 | 버전 | 출처 |
|--------|------|------|
| torch | 2.4.1+cu121 | SegEarth-OV-3 |
| mmsegmentation | 1.2.2 | SegEarth-OV-3 |
| mmengine | 0.10.7 | SegEarth-OV-3 |
| mmcv | 2.2.0 | SegEarth-OV-3 |
| PyQt5 | 5.15.11 | X-AnyLabeling |
| opencv-contrib-python-headless | 4.9.0.80 | 공통 |
| numpy | 1.26.4 | 공통 |

---

## 5. 추론 파이프라인 상세

```
[X-AnyLabeling UI] "Run" 버튼 클릭
         │
         ▼
[predict_shapes(image, filename)]
         │
         ├─ qt_img_to_rgb_cv_img() → numpy RGB (H, W, 3)
         ├─ PIL.Image.fromarray() → PIL Image
         │
         ├─ 이미지 크기 판단
         │   ├─ slide_crop < W or H → _slide_inference() (512x512 패치)
         │   └─ 그 외 → _inference_single_view()
         │
         ▼
[_inference_single_view(pil_image)]
         │
         ├─ processor.set_image(image) → ViT 백본 → 이미지 임베딩
         │
         ├─ 클래스별 루프 (13개 쿼리 워드):
         │   ├─ processor.reset_all_prompts()
         │   ├─ processor.set_text_prompt(query_word) → 텍스트 인코딩
         │   ├─ 인스턴스 헤드: masks_logits × object_score
         │   ├─ 시맨틱 헤드: semantic_mask_logits
         │   ├─ 융합: max(instance, semantic)
         │   └─ × presence_score (클래스 존재 여부)
         │
         └─ seg_logits (num_queries, H, W)
                  │
                  ▼
[후처리]
         ├─ 동의어 집계: num_queries(13) → num_cls(8)
         ├─ argmax → seg_pred (H, W) 클래스 인덱스
         ├─ prob_thd 미만 → background
         │
         ▼
[_mask_to_shapes(seg_pred)]
         │
         ├─ 클래스별 바이너리 마스크 추출
         ├─ cv2.findContours(RETR_EXTERNAL)
         ├─ refine_contours() (Douglas-Peucker 근사 + 필터링)
         └─ Shape(polygon) 객체 생성
                  │
                  ▼
[AutoLabelingResult(shapes, replace=True)]
         │
         ▼
[Canvas에 폴리곤 오버레이 표시]
```

---

## 6. 테스트 결과

### 6.1 모델 로딩 테스트

```
[STATUS] Loading SegEarth-OV-3 model... This may take a while.
[STATUS] SegEarth-OV-3 model loaded successfully.

Classes: ['background', 'bareland,barren', 'grass', 'road', 'tree,forest', 'water,river', 'cropland', 'building,roof,house']
Query words: ['background', 'bareland', 'barren', 'grass', 'road', 'tree', 'forest', 'water', 'river', 'cropland', 'building', 'roof', 'house']
num_cls=8, num_queries=13
Device: cuda
```

### 6.2 추론 테스트

- **입력 이미지**: `resources/oem_koeln_50.tif` (1000x1000 원격탐사)
- **슬라이딩 윈도우**: 512x512, stride 512

```
Result: 132 shapes detected
  bareland:  4 polygons
  building: 37 polygons
  cropland:  1 polygons
  grass:    42 polygons
  road:     10 polygons
  tree:     29 polygons
  water:     9 polygons
```

---

## 7. 실행 방법

```bash
conda activate segearth_stable
cd ~/capstone/X-AnyLabeling
python -m anylabeling.app
```

1. AI 모델 드롭다운에서 **"SegEarth-OV-3 (SAM3)"** 선택
2. 원격탐사 이미지 열기
3. **Run** 버튼 클릭
4. 클래스별 폴리곤 어노테이션 자동 생성
5. confidence 슬라이더로 필터링 조절 가능

---

## 8. 파일 구조 요약

```
X-AnyLabeling/
├── anylabeling/
│   ├── configs/
│   │   ├── auto_labeling/
│   │   │   └── segearthov3.yaml          ← [NEW] 모델 설정
│   │   └── models.yaml                   ← [EDIT] 레지스트리 등록
│   └── services/
│       └── auto_labeling/
│           ├── __init__.py               ← [EDIT] 타입 등록
│           ├── model_manager.py          ← [EDIT] 로딩 분기
│           └── segearthov3.py            ← [NEW] 래퍼 클래스
│
SegEarth-OV-3/
├── segearthov3_segmentor.py              (원본 세그멘터 - 참조)
├── sam3/                                 (SAM3 모델 코어)
│   ├── model_builder.py                  (build_sam3_image_model)
│   └── model/
│       └── sam3_image_processor.py       (Sam3Processor)
└── weights/sam3/sam3.pt                  (3.3GB 가중치)
```
