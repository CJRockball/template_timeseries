"""Strict final-evaluation specification; freeze it before holdout access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml_template.evaluation.predictor_config import PredictorConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PolicySelectionRef(StrictModel):
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_run_id: str = Field(min_length=1)
    calibration_id: str
    score_variant: Literal["raw", "calibrated_crossfit"]
    objective: Literal["f1", "f2", "manual", "alert_rate"]
    transfer_acknowledged: bool = False


class PolicyConfig(StrictModel):
    kind: Literal["none", "threshold", "alert_rate"] = "none"
    value: float | None = None
    source: str = "No policy selected"
    batch_definition: str = "entire evaluation holdout"
    selection: PolicySelectionRef | None = None

    @model_validator(mode="after")
    def check(self):
        if self.kind == "none" and self.value is not None:
            raise ValueError("none policy cannot have a value")
        if self.kind != "none" and (self.value is None or not 0 <= self.value <= 1):
            raise ValueError("policy value must be in [0,1]")
        if self.kind == "alert_rate" and self.value == 0:
            raise ValueError("alert rate must be positive")
        if self.kind == "none" and self.selection is not None:
            raise ValueError("none policy cannot have selection provenance")
        if not self.batch_definition.strip():
            raise ValueError("document the policy batch definition")
        if not self.source.strip():
            raise ValueError("document the policy source")
        return self


class UncertaintyConfig(StrictModel):
    enabled: bool = True
    repetitions: int = Field(default=1000, ge=100, le=20000)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    sampling_unit: Literal["independent_row"] = "independent_row"


class SliceConfig(StrictModel):
    column: str
    values: tuple[str | int | float | bool, ...]

    @model_validator(mode="after")
    def check(self):
        if not self.column.strip() or not self.values:
            raise ValueError("slice needs an input-feature column and explicit values")
        return self


class Criterion(StrictModel):
    metric: str
    operator: Literal[">=", "<="]
    value: float
    basis: Literal["point", "confidence_bound"] = "point"


class EvaluationConfig(StrictModel):
    schema_version: Literal[1] = 1
    parent_run_id: str = Field(min_length=1)
    calibration_id: str | None = None
    release_variant: Literal["raw", "calibrated"] = "raw"
    expected_silver_dataset_version_id: str | None = None
    expected_feature_set_id: str | None = None
    expected_split_id: str | None = None
    intended_use: str = "Research; no automatic deployment approval"
    limitations: tuple[str, ...] = ()
    primary_metric: Literal["pr_auc", "roc_auc", "brier", "log_loss"] = "pr_auc"
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    alert_rates: tuple[float, ...] = (0.05, 0.10)
    reliability_bins: int = Field(default=5, ge=2, le=20)
    slices: tuple[SliceConfig, ...] = ()
    min_slice_rows: int = Field(default=30, ge=1)
    min_slice_positives: int = Field(default=10, ge=1)
    error_sample_size: int = Field(default=25, ge=0, le=1000)
    criteria: tuple[Criterion, ...] = ()
    acknowledge_prior_exposure: bool = False
    prior_exposure_note: str = ""
    supersedes_evaluation_id: str | None = None
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)

    @model_validator(mode="after")
    def check(self):
        if self.predictor.kind == "cv_mean" and (
            self.calibration_id is not None
            or self.release_variant != "raw"
            or self.policy.kind != "none"
        ):
            raise ValueError(
                "cv_mean initially supports raw probabilities with no calibrator or decision policy"
            )
        if self.release_variant == "calibrated" and not self.calibration_id:
            raise ValueError("calibrated release requires calibration_id")
        ref = self.policy.selection
        if ref is not None:
            if ref.parent_run_id != self.parent_run_id or ref.calibration_id != self.calibration_id:
                raise ValueError("policy-selection predictor identity mismatch")
            expected = "calibrated_crossfit" if self.release_variant == "calibrated" else "raw"
            if ref.score_variant != expected:
                raise ValueError("policy-selection score scale mismatch")
            if expected == "calibrated_crossfit" and not ref.transfer_acknowledged:
                raise ValueError("acknowledge cross-fit-to-final calibrator transfer")
        if len(set(self.alert_rates)) != len(self.alert_rates) or any(
            not 0 < r <= 1 for r in self.alert_rates
        ):
            raise ValueError("rates must be unique and in (0,1]")
        if len({s.column for s in self.slices}) != len(self.slices):
            raise ValueError("declare each slice once")
        if self.acknowledge_prior_exposure and not self.prior_exposure_note.strip():
            raise ValueError("acknowledgement requires a note")
        return self

    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_evaluation_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("task") == "regression":
        from ml_template.evaluation.regression_config import RegressionEvaluationConfig

        return RegressionEvaluationConfig.model_validate(raw)
    return EvaluationConfig.model_validate(raw)


# calibrated-policy update v1
