"""Configuration for post-selection calibration, separate from RunConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    parent_run_id: str = Field(pattern=r"^run_[A-Za-z0-9_-]+$")
    method: Literal["none", "sigmoid"] = "sigmoid"
    epsilon: float = Field(1e-7, gt=0, lt=0.01, allow_inf_nan=False)
    l2: float = Field(1e-6, ge=0, allow_inf_nan=False)
    reliability_bins: int = Field(5, ge=2, le=50)
    min_class_count: int = Field(10, ge=2)


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("calibration YAML must be a mapping")
    return CalibrationConfig.model_validate(raw)
