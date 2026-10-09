from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44_100
    frame_length: int = 1_024
    hop_length: int = 512
    n_fft: int = 1_024
    n_mels: int = 40
    n_mfcc: int = 25
    fmin: float = 0.0
    fmax: float = 22_050.0
    feature_variant: str = "paper225"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FeatureConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in fields if key in data})


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config
