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

### `rcr/__init__.py`

Exports the RCR inferencer and config loader.

### `tfcc/`

Provides shared class-detail extraction utilities used by RCR.

RCR needs class-level raw logits, semantic logits, instance logits, head agreement, and presence values. These are produced by `tfcc/tfcc_inferencer.py` without model training or weight updates.

Key files:

- `tfcc/tfcc_inferencer.py`: extracts class-wise SegEarth-OV3 evidence.
- `tfcc/transforms.py`: applies and inverses TTA views.
- `tfcc/mask_utils.py`: small mask utility functions.
- `tfcc/config.py`: config structure used by the shared inferencer.

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

## Notes For Reuse

- RCR-SegEarth does not call `train()`, `backward()`, or `optimizer.step()`.
- SAM3 and SegEarth-OV3 weights remain unchanged.
- The baseline path remains available through the original configs.
- RCR is intended as a conservative refinement module, so large changes usually indicate an overly aggressive YAML setting.
- For paper tables, use the best threshold setting per dataset and report both the summary metrics and `class_iou_results.csv`.
