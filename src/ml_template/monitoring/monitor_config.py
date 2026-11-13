"""Strict monitoring requests. Paths are relative to the invocation directory."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1


class ReferenceConfig(StrictRequest):
    scoring_batch_id: str
    name: str = Field(min_length=1)
    approved_population: str = Field(min_length=1)
    numeric_bins: int = Field(default=10, ge=2, le=50)
    target_unit: str | None = Field(default=None, min_length=1)


class LabelConfig(StrictRequest):
    scoring_batch_id: str
    input_path: Path
    source: str = Field(min_length=1)
    outcome_definition: str = Field(min_length=1)
    sampling: Literal["all_rows", "selected", "unknown"]
    supersedes_label_import_id: str | None = None
    target_unit: str | None = Field(default=None, min_length=1)


class MonitorConfig(StrictRequest):
    scoring_batch_id: str
    reference_id: str
    label_import_id: str | None = None
    as_of: datetime
    min_rows: int = Field(default=200, ge=1)
    min_labelled: int = Field(default=100, ge=1)
    min_positives: int = Field(default=10, ge=1)
    min_negatives: int = Field(default=10, ge=1)
    js_warning: float = Field(default=0.1, ge=0, le=1)
    min_label_coverage: float = Field(default=0.9, ge=0, le=1)
    brier_warning: float | None = Field(default=None, ge=0, le=1)
    recall_floor: float | None = Field(default=None, ge=0, le=1)
    quality_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    target_unit: str | None = Field(default=None, min_length=1)
    rmse_warning: float | None = Field(default=None, ge=0)
    mae_warning: float | None = Field(default=None, ge=0)
    abs_bias_warning: float | None = Field(default=None, ge=0)
    missing_rate_warning: float = Field(default=1.0, ge=0, le=1)
    missing_rate_delta_warning: float = Field(default=0.1, ge=0, le=1)
    interval_coverage_slack: float = Field(default=0.05, ge=0, le=1)

    @model_validator(mode="after")
    def validate_options(self):
        if self.as_of.tzinfo is None:
            raise ValueError("as_of requires an explicit timezone")
        if any(low >= high for low, high in self.quality_ranges.values()):
            raise ValueError("range lower bound must be below upper bound")
        return self


def load_request(path, cls):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("request must be a YAML mapping")
    return cls.model_validate(raw)
