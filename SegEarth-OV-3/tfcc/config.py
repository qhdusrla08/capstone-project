from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class TFCCWeights:
    head_agreement: float = 0.12
    geometric: float = 0.10


@dataclass
class TFCCThresholds:
    mask_threshold: float | None = None
    exemplar_presence: float = 0.45
    exemplar_mask_score: float = 0.45
    exemplar_head_iou: float = 0.35
    uncertain_low: float = 0.20
    uncertain_high: float = 0.72


@dataclass
class TFCCGeometry:
    enabled: bool = True
    presence_top_k: int = 4
    max_geo_classes: int = 2
    views: list[str] = field(default_factory=lambda: ["hflip", "rot90"])
    variance_weight: float = 0.30


@dataclass
class TFCCTTA:
    enabled: bool = False
    views: list[str] = field(default_factory=lambda: ["hflip"])
    fuse_weight: float = 0.50


@dataclass
class TFCCDenseHead:
    enabled: bool = False
    consensus_weight: float = 0.08
    disagreement_weight: float = 0.03


@dataclass
class TFCCOnlineOptimization:
    enabled: bool = False
    steps: int = 3
    lr: float = 0.08
    weight_decay: float = 0.0
    temperature: float = 1.0
    entropy_weight: float = 0.01
    bias_l2_weight: float = 0.02
    max_abs_bias: float = 0.12
    presence_top_k: int = 4
    max_classes: int = 4
    pixel_margin: float = 0.12
    min_confidence: float | None = None


@dataclass
class TFCCCalibration:
    enabled: bool = True
    presence_top_k: int = 4
    max_classes: int = 4
    pixel_support_threshold: float | None = None
    max_margin: float = 0.08
    max_changed_ratio: float = 0.02


@dataclass
class TFCCOutput:
    save_json: bool = False
    keep_logits: bool = True
    keep_dense_results: bool = True
    save_raw_result: bool = True
    save_calibrated_result: bool = True
    save_dense_arrays: bool = False


@dataclass
class TFCCConfig:
    use_tfcc: bool = True
    zero_shot_mode: bool = True
    bg_idx: int | None = None
    prob_threshold: float | None = None
    exclude_classes: list[str] = field(default_factory=lambda: ["background", "background class"])
    weights: TFCCWeights = field(default_factory=TFCCWeights)
    thresholds: TFCCThresholds = field(default_factory=TFCCThresholds)
    geometry: TFCCGeometry = field(default_factory=TFCCGeometry)
    tta: TFCCTTA = field(default_factory=TFCCTTA)
    dense_head: TFCCDenseHead = field(default_factory=TFCCDenseHead)
    online: TFCCOnlineOptimization = field(default_factory=TFCCOnlineOptimization)
    calibration: TFCCCalibration = field(default_factory=TFCCCalibration)
    output: TFCCOutput = field(default_factory=TFCCOutput)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TFCCConfig":
        if data is None:
            return cls()
        return _dataclass_from_mapping(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TFCCConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"TFCC config must be a mapping: {config_path}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tfcc_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> TFCCConfig:
    config = TFCCConfig.from_yaml(path) if path else TFCCConfig()
    if overrides:
        config = TFCCConfig.from_dict(_deep_update(config.to_dict(), overrides))
    return config


def _dataclass_from_mapping(dataclass_type: type[Any], data: Mapping[str, Any]) -> Any:
    allowed_fields = {item.name: item for item in fields(dataclass_type)}
    unknown_keys = sorted(set(data) - set(allowed_fields))
    if unknown_keys:
        raise ValueError(f"Unknown TFCC config keys for {dataclass_type.__name__}: {unknown_keys}")

    values: dict[str, Any] = {}
    defaults = dataclass_type()
    for name, item in allowed_fields.items():
        if name not in data:
            continue
        default_value = getattr(defaults, name)
        raw_value = data[name]
        if is_dataclass(default_value) and isinstance(raw_value, Mapping):
            values[name] = _dataclass_from_mapping(type(default_value), raw_value)
        else:
            values[name] = raw_value
    return dataclass_type(**values)


def _deep_update(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
