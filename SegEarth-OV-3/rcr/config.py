from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class RCRTTA:
    enabled: bool = True
    views: list[str] = field(default_factory=lambda: ["hflip"])
    fuse_weight: float = 0.55
    use_consensus_as_base: bool = True


@dataclass
class RCRBoundary:
    enabled: bool = True
    kernel_size: int = 3
    max_margin: float = 0.08
    min_consensus_gain: float = -0.02
    label_boost: float = 0.12
    high_confidence_lock: float = 0.85
    lock_margin: float = 0.18
    allow_background_relabel: bool = False
    max_changed_ratio: float = 0.06


@dataclass
class RCRComponent:
    enabled: bool = True
    min_area: int = 32
    min_area_ratio: float = 0.00003
    protect_confidence: float = 0.70
    neighbor_kernel_size: int = 5
    label_boost: float = 0.10
    allow_background_relabel: bool = True


@dataclass
class RCRDenseHead:
    enabled: bool = True
    consensus_weight: float = 0.05
    disagreement_weight: float = 0.02
    max_margin: float = 0.10


@dataclass
class RCRLocalVote:
    enabled: bool = True
    kernel_size: int = 5
    min_vote_ratio: float = 0.55
    max_margin: float = 0.14
    consensus_agree_required: bool = False
    min_consensus_support: float = -0.04
    label_boost: float = 0.14
    allow_background_relabel: bool = False
    max_changed_ratio: float = 0.08


@dataclass
class RCRSafety:
    max_total_changed_ratio: float = 0.08


@dataclass
class RCROutput:
    save_json: bool = False
    keep_logits: bool = True
    save_masks: bool = False


@dataclass
class RCRConfig:
    use_rcr: bool = True
    bg_idx: int | None = None
    prob_threshold: float | None = None
    exclude_classes: list[str] = field(default_factory=lambda: ["background", "background class"])
    tta: RCRTTA = field(default_factory=RCRTTA)
    boundary: RCRBoundary = field(default_factory=RCRBoundary)
    component: RCRComponent = field(default_factory=RCRComponent)
    dense_head: RCRDenseHead = field(default_factory=RCRDenseHead)
    local_vote: RCRLocalVote = field(default_factory=RCRLocalVote)
    safety: RCRSafety = field(default_factory=RCRSafety)
    output: RCROutput = field(default_factory=RCROutput)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RCRConfig":
        if data is None:
            return cls()
        return _dataclass_from_mapping(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RCRConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"RCR config must be a mapping: {config_path}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rcr_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> RCRConfig:
    config = RCRConfig.from_yaml(path) if path else RCRConfig()
    if overrides:
        config = RCRConfig.from_dict(_deep_update(config.to_dict(), overrides))
    return config


def _dataclass_from_mapping(dataclass_type: type[Any], data: Mapping[str, Any]) -> Any:
    allowed_fields = {item.name: item for item in fields(dataclass_type)}
    unknown_keys = sorted(set(data) - set(allowed_fields))
    if unknown_keys:
        raise ValueError(f"Unknown RCR config keys for {dataclass_type.__name__}: {unknown_keys}")

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
