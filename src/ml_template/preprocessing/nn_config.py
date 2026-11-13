"""Strict, model-independent configuration for saved NN input preprocessing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NNPreprocessorConfig:
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    numeric_strategy: str = "median"
    scaling: str = "standard"
    constant_fill: float = 0.0
    all_missing_fill: float = 0.0
    unknown_policy: str = "reserved"

    def __post_init__(self) -> None:
        for name in ("numeric", "categorical"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{name} must be an explicit list of column names")
            if any(not isinstance(c, str) or not c for c in values):
                raise ValueError(f"{name} requires nonempty string column names")
            object.__setattr__(self, name, tuple(values))
        columns = self.numeric + self.categorical
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(
                "select at least one input; numeric/categorical lists must be disjoint"
            )
        outputs = (*columns, *(f"{c}__is_missing" for c in columns))
        if len(set(outputs)) != len(outputs):
            raise ValueError("input names collide with generated missing flags")
        if self.numeric_strategy not in ("mean", "median", "constant"):
            raise ValueError("numeric_strategy must be mean, median or constant")
        if self.scaling not in ("standard", "none"):
            raise ValueError("scaling must be standard or none")
        if self.unknown_policy not in ("reserved", "error"):
            raise ValueError("unknown_policy must be reserved or error")
        for name in ("constant_fill", "all_missing_fill"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))

    @property
    def input_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["numeric"] = list(self.numeric)
        result["categorical"] = list(self.categorical)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NNPreprocessorConfig:
        if not isinstance(raw, dict):
            raise ValueError("preprocessing configuration must be a mapping")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown preprocessing settings: {sorted(unknown)}")
        return cls(**raw)


def load_nn_preprocessor_config(path: str | Path) -> NNPreprocessorConfig:
    """Load a dedicated preprocessing YAML, not an experiment YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        return NNPreprocessorConfig.from_dict(yaml.safe_load(handle))
