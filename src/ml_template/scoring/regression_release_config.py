"""Explicit research-only continuous-prediction release request."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegressionReleaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[2] = 2
    task: Literal["regression"] = "regression"
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    evaluation_id: str = Field(min_length=1)
    parent_run_id: str = Field(min_length=1)
    calibration_id: str | None = None
    intended_use: str = Field(min_length=1)
    acknowledge_research_release: bool = False
    parity_atol: float = Field(default=1e-7, gt=0, le=1e-5)

    @model_validator(mode="after")
    def _check(self):
        if not self.acknowledge_research_release:
            raise ValueError("Explicit research-release acknowledgement required")
        return self
