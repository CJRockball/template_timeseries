"""Frozen point-estimate regression evaluation specification."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from ml_template.evaluation.predictor_config import PredictorConfig


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DisabledUncertainty(Strict):
    enabled: bool = False
    coverages: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _levels(self):
        from ml_template.models.regression_postprocessing import levels

        levels(self.coverages)
        if bool(self.coverages) != self.enabled:
            raise ValueError(
                "enabled intervals require explicit levels; disabled intervals require none"
            )
        return self

    @model_serializer(mode="wrap")
    def _legacy(self, handler):
        data = handler(self)
        if not self.coverages:
            data.pop("coverages", None)
        return data


class RegressionCriterion(Strict):
    metric: Literal["rmse", "mae", "r2"]
    operator: Literal["<=", ">="]
    value: float
    basis: Literal["point"] = "point"


class RegressionEvaluationConfig(Strict):
    schema_version: Literal[2] = 2
    task: Literal["regression"] = "regression"
    parent_run_id: str = Field(min_length=1)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    primary_metric: Literal["rmse", "mae"] = "rmse"
    expected_silver_dataset_version_id: str | None = None
    expected_feature_set_id: str | None = None
    expected_split_id: str | None = None
    intended_use: str = "Research; no automatic deployment approval"
    limitations: tuple[str, ...] = ()
    criteria: tuple[RegressionCriterion, ...] = ()
    uncertainty: DisabledUncertainty = Field(default_factory=DisabledUncertainty)
    calibration_id: str | None = None
    release_variant: Literal["raw", "recalibrated"] = "raw"
    target_unit: str | None = None

    @model_serializer(mode="wrap")
    def _legacy(self, handler):
        data = handler(self)
        if self.target_unit is None:
            data.pop("target_unit", None)
        return data

    acknowledge_prior_exposure: bool = False
    prior_exposure_note: str = ""
    supersedes_evaluation_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.calibration_id is None and (
            self.uncertainty.enabled or self.release_variant != "raw"
        ):
            raise ValueError("intervals/correction require an explicit calibration ID")
        if self.calibration_id is not None and (
            not self.calibration_id.startswith("regcal_")
            or not self.target_unit
            or not self.target_unit.strip()
        ):
            raise ValueError("explicit regression calibration ID and target units required")
        if not self.intended_use.strip():
            raise ValueError("Document intended use")
        if self.acknowledge_prior_exposure and not self.prior_exposure_note.strip():
            raise ValueError("Prior-exposure acknowledgement requires a note")
        return self
