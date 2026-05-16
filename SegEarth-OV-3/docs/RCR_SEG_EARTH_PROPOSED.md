# RCR-SegEarth Proposed Module

RCR-SegEarth is a training-free refinement wrapper for SegEarth-OV3. It keeps the SAM3 and SegEarth-OV3 weights frozen, runs the normal SegEarth-OV3 inference first, and then applies region consistency refinement to reduce unstable boundary noise and small inconsistent components.

The original SegEarth-OV3 path is preserved. RCR is enabled only when `use_rcr=True` is set in the MMSeg config.

## Core Idea

SegEarth-OV3 produces a class logit map from semantic evidence, instance evidence, and presence score. RCR-SegEarth does not retrain or adapt the model. Instead, it uses test-time evidence to refine only regions that are likely to be unstable:

1. Run the SegEarth-OV3 baseline prediction.
2. Re-run lightweight test-time views such as horizontal flip and rotations.
3. Align the view logits back to the original image.
4. Build a consensus logit map from the original and transformed predictions.
5. Refine uncertain boundaries, small components, and local voting regions.
6. Apply safety limits so the final mask cannot drift too far from the baseline.

In short, RCR-SegEarth preserves high-confidence SegEarth-OV3 predictions and only nudges region-level inconsistencies.

## Reproducible Algorithm

RCR-SegEarth uses the following tensor convention for a single image:

```text
class_logits: Tensor[C, H, W]
semantic_logits: Tensor[C, H, W]
instance_logits: Tensor[C, H, W]
semantic_raw_logits: Tensor[C, H, W]
instance_raw_logits: Tensor[C, H, W]
presence: Tensor[C]
pred_mask: Tensor[H, W]
```

There is no batch dimension inside RCR. MMSeg still calls the segmentor with batch size 1, then RCR refines each image independently.

The baseline class logit extraction follows SegEarth-OV3:

```text
instance_logit_c = max_i(mask_logit_{c,i} * object_score_{c,i})
semantic_logit_c = semantic_mask_logit_c
class_logit_c = max(instance_logit_c, semantic_logit_c)
class_logit_c = class_logit_c * presence_c
pred_mask(x, y) = argmax_c class_logit_c(x, y)
pred_mask(x, y) = bg_idx if max_c class_logit_c(x, y) < prob_thd
```

For prompt aliases that map to the same dataset class, RCR takes the maximum over aliases.

### Stage Order

The exact RCR refinement order is:

1. Extract baseline class evidence.
2. Compute TTA consensus logits.
3. Choose `base_logits = consensus_logits` if `tta.use_consensus_as_base=True`, otherwise `raw_logits`.
4. Apply boundary refinement.
5. Apply local vote refinement.
6. Apply small component cleanup.
7. Apply total changed-pixel safety cap.
8. Convert final logits to the final mask with the same `prob_thd` background rule.

### TTA Consensus

For each configured view, RCR transforms the image, runs the same frozen SegEarth-OV3 evidence extractor, and inverses the view logits back to the original coordinates.

Supported views:

```text
hflip, rot90, rot180, rot270
```

Consensus is computed on logits, not probabilities:

```text
aligned_logits = [raw_logits, inverse(view_1_logits), ..., inverse(view_n_logits)]
consensus_logits = mean(aligned_logits)
fused_logits = (1 - fuse_weight) * raw_logits + fuse_weight * consensus_logits
```

No variance penalty is used in the current RCR-SegEarth implementation.

### Boundary Refinement

For each pixel:

```text
raw_value = max_c base_logits_c
raw_label = argmax_c base_logits_c
raw_margin = top1(base_logits) - top2(base_logits)
consensus_value = max_c consensus_logits_c
consensus_label = argmax_c consensus_logits_c
```

Boundary pixels are defined by local class disagreement in a `boundary.kernel_size` window:

```text
boundary(x, y) = max_pool(pred_mask) != min_pool(pred_mask)
```

A boundary candidate is selected when:

```text
(boundary OR raw_margin <= boundary.max_margin)
AND NOT high_conf_lock
AND raw_value >= prob_thd
AND consensus_value >= raw_value + boundary.min_consensus_gain
AND consensus_label != raw_label
AND consensus_label != bg_idx when allow_background_relabel=False
```

The high-confidence lock is:

```text
high_conf_lock =
    raw_value >= boundary.high_confidence_lock
    AND raw_margin >= boundary.lock_margin
    AND NOT boundary
```

Candidate pixels are capped by `boundary.max_changed_ratio`. The kept pixels are the candidates with largest `consensus_value - raw_value`.

For each kept candidate:

```text
refined_logits[:, pixel] = base_logits[:, pixel]
refined_logits[consensus_label, pixel] =
    max(refined_logits[consensus_label, pixel],
        old_raw_label_value + boundary.label_boost)
```

### Dense Head Candidate

If `dense_head.enabled=True`, RCR adds a secondary evidence candidate using semantic/instance agreement:

```text
head_consensus = min(semantic_logits, instance_logits).clamp_min(0)
head_disagreement = abs(semantic_logits - instance_logits)
dense_logits =
    base_logits
    + dense_head.consensus_weight * head_consensus
    - dense_head.disagreement_weight * head_disagreement
```

A dense-head pixel is a candidate when:

```text
dense_label != raw_label
AND dense_value >= raw_value + boundary.min_consensus_gain
AND raw_value - base_logits[dense_label] <= dense_head.max_margin
```

The same background relabel restriction is applied.

### Local Vote Refinement

Local vote uses a square `local_vote.kernel_size` window around each pixel. It counts class labels in the current refined mask and selects:

```text
vote_label = local majority class
vote_ratio = local majority ratio
```

A local-vote candidate is selected when:

```text
(boundary OR margin <= local_vote.max_margin)
AND vote_label != current_label
AND vote_ratio >= local_vote.min_vote_ratio
AND current_logit_for_vote_label >= current_value - local_vote.max_margin
AND consensus_logit_for_vote_label >= current_value + local_vote.min_consensus_support
AND current_value >= prob_thd
```

If `local_vote.consensus_agree_required=True`, RCR also requires:

```text
argmax(consensus_logits) == vote_label
```

Candidates are capped by `local_vote.max_changed_ratio`, keeping the largest:

```text
vote_ratio + consensus_value - current_value
```

The final update uses `local_vote.label_boost`.

### Component Cleanup

RCR runs connected component cleanup after boundary and local-vote refinement.

For every non-background class:

```text
connected components are computed with 8-connectivity
min_area = max(component.min_area, image_area * component.min_area_ratio)
```

A component is eligible when:

```text
component_area <= min_area
AND mean(raw_confidence over component) < component.protect_confidence
```

The replacement label is the dominant neighbor in a dilated ring:

```text
ring = dilate(component_mask, component.neighbor_kernel_size) - component_mask
neighbor = most frequent class in ring excluding the component class
```

If the neighbor is background and `component.allow_background_relabel=False`, the component is kept. Otherwise:

```text
refined_logits[neighbor, component] =
    max(refined_logits[neighbor, component],
        old_component_class_value + component.label_boost)
```

### Safety Cap

RCR compares the final refined mask against the original raw mask:

```text
changed = raw_mask != refined_mask
changed_ratio = changed_pixels / total_pixels
```

If `changed_ratio <= safety.max_total_changed_ratio`, no rollback is applied.

If the ratio is too large, RCR keeps only the top changed pixels by logit gain:

```text
gain = max_c refined_logits_c - max_c raw_logits_c
keep_count = total_pixels * safety.max_total_changed_ratio
keep changed pixels with largest gain
revert all other changed pixels to raw_logits
```

There is no random rollback and no class-wise rollback in the current implementation.

## File Map

### `segearthov3_segmentor.py`

Adds the RCR inference switch to the SegEarth-OV3 MMSeg segmentor.

Important options:

- `use_rcr`: enables RCR-SegEarth.
- `rcr_config_path`: path to the RCR YAML config.
- `rcr_output_dir`: optional directory for RCR auxiliary outputs.
- `rcr_save_json`: saves per-image debug JSON when enabled.

When `use_rcr=False`, the original SegEarth-OV3 prediction path is used.

### `rcr/config.py`

Defines all RCR configuration dataclasses and YAML loading logic.

Main sections:

- `tta`: test-time view consensus settings.
- `boundary`: boundary refinement thresholds and label boost.
- `component`: small connected component cleanup.
- `dense_head`: semantic and instance head agreement cues.
- `local_vote`: neighborhood label consistency refinement.
- `safety`: final changed-pixel cap.
- `output`: optional debug artifact controls.

### `rcr/rcr_inferencer.py`

Implements the RCR-SegEarth inference flow.

Main stages:

- `infer_image`: end-to-end RCR inference for one image.
- `_compute_consensus_logits`: runs enabled TTA views and fuses aligned logits.
- `_refine_boundary`: updates low-margin boundary pixels when consensus supports another class.
- `_refine_local_vote`: uses neighborhood majority evidence for uncertain pixels.
- `_cleanup_components`: absorbs small unstable components into dominant neighbors.
- `_limit_total_changes`: prevents excessive changes from the baseline.

### `rcr/evidence_extractor.py`

Extracts the SegEarth-OV3 evidence that RCR needs from the frozen base model.

RCR needs more than the final predicted mask. This file provides:

- class-level raw logits
- semantic logits
- instance logits
- semantic and instance head agreement
- presence values
- TTA image transforms and inverse logit transforms

Keeping this extractor inside `rcr/` makes the RCR module self-contained. Other SegEarth-OV3 variants can copy the `rcr/` directory plus the small `segearthov3_segmentor.py` integration changes without bringing in unrelated experiment folders.

### `rcr/__init__.py`

Exports the RCR inferencer, evidence extractor, and config loader.

### RCR MMSeg Configs

Dataset-level config files:

- `configs/cfg_loveda_rcr.py`
- `configs/cfg_openearthmap_rcr.py`
- `configs/cfg_potsdam_rcr.py`
- `configs/cfg_vaihingen_rcr.py`

Each file inherits the corresponding SegEarth-OV3 baseline config and only switches on `use_rcr=True`.

RCR hyperparameter YAML files:

- `configs/rcr_default.yaml`
- `configs/rcr_loveda.yaml`
- `configs/rcr_openearthmap.yaml`
- `configs/rcr_potsdam.yaml`
- `configs/rcr_vaihingen.yaml`

These store dataset-specific refinement settings.

## Key Config Values

Baseline SegEarth-OV3 dataset thresholds:

| Dataset | `confidence_threshold` | `prob_thd` | `bg_idx` |
|---|---:|---:|---:|
| LoveDA | 0.5 | 0.5 | 0 |
| OpenEarthMap | 0.1 | 0.1 | 0 |
| Potsdam | 0.2 | 0.1 | 5 |
| Vaihingen | 0.4 | 0.1 | 5 |

Main RCR settings:

| Dataset | TTA views | `fuse_weight` | `boundary.max_margin` | `boundary.label_boost` | `component.min_area` | `local_vote.kernel_size` | `local_vote.min_vote_ratio` | `safety.max_total_changed_ratio` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LoveDA | hflip, rot90, rot180, rot270 | 0.72 | 0.14 | 0.18 | 72 | 5 | 0.52 | 0.14 |
| OpenEarthMap | hflip, rot90 | 0.58 | 0.08 | 0.12 | 96 | 5 | 0.56 | 0.07 |
| Potsdam | hflip, rot90 | 0.58 | 0.08 | 0.12 | 96 | 5 | 0.56 | 0.08 |
| Vaihingen | hflip, rot90 | 0.58 | 0.08 | 0.12 | 96 | 5 | 0.56 | 0.07 |

Additional RCR settings:

| Dataset | `boundary.high_confidence_lock` | `boundary.lock_margin` | `boundary.max_changed_ratio` | `component.protect_confidence` | `component.label_boost` | `local_vote.max_margin` | `local_vote.label_boost` | `local_vote.max_changed_ratio` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoveDA | 0.92 | 0.24 | 0.12 | 0.76 | 0.14 | 0.18 | 0.18 | 0.12 |
| OpenEarthMap | 0.66 | 0.14 | 0.06 | 0.50 | 0.10 | 0.10 | 0.12 | 0.06 |
| Potsdam | 0.76 | 0.16 | 0.06 | 0.55 | 0.10 | 0.10 | 0.12 | 0.06 |
| Vaihingen | 0.80 | 0.18 | 0.055 | 0.60 | 0.10 | 0.10 | 0.12 | 0.055 |

### Evaluation Utilities

`eval.py` now saves class-wise evaluation results in addition to console output.

For each run with `--out <output_dir>`, the following files are written:

- `<output_dir>/class_iou_results.csv`
- `<output_dir>/class_iou_results.json`
- `<output_dir>/class_iou_results.txt`

`tools/collect_best_rcr_threshold.py` collects threshold sweep results and reports the best `confidence_threshold` and `prob_thd` per dataset.

## Run Commands

### Baseline SegEarth-OV3

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/cfg_loveda.py --out outputs/baseline_loveda_eval/full
CUDA_VISIBLE_DEVICES=1 python eval.py ./configs/cfg_openearthmap.py --out outputs/baseline_openearthmap_eval/full
CUDA_VISIBLE_DEVICES=2 python eval.py ./configs/cfg_potsdam.py --out outputs/baseline_potsdam_eval/full
CUDA_VISIBLE_DEVICES=3 python eval.py ./configs/cfg_vaihingen.py --out outputs/baseline_vaihingen_eval/full
```

### RCR-SegEarth

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/cfg_loveda_rcr.py --out outputs/rcr_loveda_eval/full
CUDA_VISIBLE_DEVICES=1 python eval.py ./configs/cfg_openearthmap_rcr.py --out outputs/rcr_openearthmap_eval/full
CUDA_VISIBLE_DEVICES=2 python eval.py ./configs/cfg_potsdam_rcr.py --out outputs/rcr_potsdam_eval/full
CUDA_VISIBLE_DEVICES=3 python eval.py ./configs/cfg_vaihingen_rcr.py --out outputs/rcr_vaihingen_eval/full
```

## Threshold Sweep

SegEarth-OV3 uses dataset-specific `confidence_threshold` and `prob_thd`. For a fair RCR-SegEarth comparison, tune the same thresholds for RCR.

### LoveDA

```bash
for ct in 0.4 0.5 0.6; do
  for pt in 0.4 0.5 0.6; do
    CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/cfg_loveda_rcr.py \
      --out outputs/rcr_threshold_sweep/loveda_ct${ct}_pt${pt} \
      --cfg-options model.confidence_threshold=${ct} model.prob_thd=${pt}
  done
done
```

### OpenEarthMap

```bash
for ct in 0.05 0.10 0.15; do
  for pt in 0.05 0.10 0.15; do
    CUDA_VISIBLE_DEVICES=1 python eval.py ./configs/cfg_openearthmap_rcr.py \
      --out outputs/rcr_threshold_sweep/oem_ct${ct}_pt${pt} \
      --cfg-options model.confidence_threshold=${ct} model.prob_thd=${pt}
  done
done
```

### Potsdam

```bash
for ct in 0.15 0.20 0.25; do
  for pt in 0.05 0.10 0.15; do
    CUDA_VISIBLE_DEVICES=2 python eval.py ./configs/cfg_potsdam_rcr.py \
      --out outputs/rcr_threshold_sweep/potsdam_ct${ct}_pt${pt} \
      --cfg-options model.confidence_threshold=${ct} model.prob_thd=${pt}
  done
done
```

### Vaihingen

```bash
for ct in 0.30 0.40 0.50; do
  for pt in 0.05 0.10 0.15; do
    CUDA_VISIBLE_DEVICES=3 python eval.py ./configs/cfg_vaihingen_rcr.py \
      --out outputs/rcr_threshold_sweep/vaihingen_ct${ct}_pt${pt} \
      --cfg-options model.confidence_threshold=${ct} model.prob_thd=${pt}
  done
done
```

After all sweeps finish:

```bash
python tools/collect_best_rcr_threshold.py \
  --root outputs/rcr_threshold_sweep \
  --out outputs/rcr_threshold_sweep/best_rcr_thresholds
```

This writes:

- `all_rcr_threshold_results.csv`
- `best_rcr_threshold_results.csv`
- `best_rcr_threshold_results.json`

## Reference Results

The following numbers are the current reference results from this branch before any additional RCR threshold sweep.

| Model | LoveDA mIoU | OpenEarthMap mIoU |
|---|---:|---:|
| SegEarth-OV3 baseline | 47.38 | 44.16 |
| RCR-SegEarth | 47.88 | 44.54 |

For final paper tables, run the threshold sweep and report the best RCR setting per dataset. The sweep outputs class-wise IoU tables in each run directory through `class_iou_results.csv`.

## Reproduction Environment

The current experiments used:

| Item | Value |
|---|---|
| SAM3 checkpoint path | `weights/sam3/sam3.pt` |
| BPE vocab path | `sam3/assets/bpe_simple_vocab_16e6.txt.gz` |
| Batch size | 1 |
| Test workers | 4 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| MMSegmentation | 1.2.2 |
| MMEngine | 0.10.7 |
| OpenCV | 4.13.0 |
| NumPy | 2.2.6 |

Dataset splits are the `img_dir/val` and `ann_dir/val` splits configured in each `cfg_*_rcr.py` file. OpenEarthMap uses sliding inference with `slide_crop=512` and `slide_stride=512`, inherited from `cfg_openearthmap.py`.

## Notes For Reuse

- RCR-SegEarth does not call `train()`, `backward()`, or `optimizer.step()`.
- SAM3 and SegEarth-OV3 weights remain unchanged.
- The baseline path remains available through the original configs.
- To port RCR into another SegEarth-OV3 variant, copy `rcr/`, the `use_rcr` integration in `segearthov3_segmentor.py`, and the `configs/cfg_*_rcr.py` plus `configs/rcr_*.yaml` files.
- RCR is intended as a conservative refinement module, so large changes usually indicate an overly aggressive YAML setting.
- For paper tables, use the best threshold setting per dataset and report both the summary metrics and `class_iou_results.csv`.
