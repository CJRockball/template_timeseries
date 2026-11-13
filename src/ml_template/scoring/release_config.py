"""Explicit release requests. Frozen policy comes from a completed evaluation."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReleaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: int = Field(default=1, ge=1, le=1)
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    evaluation_id: str = Field(min_length=1)
    parent_run_id: str = Field(min_length=1)
    calibration_id: str | None = None
    intended_use: str = Field(min_length=1)
    acknowledge_research_release: bool = False
    parity_atol: float = Field(default=1e-7, gt=0, le=1e-5)

    @model_validator(mode="after")
    def check(self):
        if not self.acknowledge_research_release:
            raise ValueError(
                "explicit research-release acknowledgement required; no deployment approval is implied"
            )
        return self


def load_release_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("task") == "regression":
        from ml_template.scoring.regression_release_config import RegressionReleaseConfig

        return RegressionReleaseConfig.model_validate(raw)
    return ReleaseConfig.model_validate(raw)
