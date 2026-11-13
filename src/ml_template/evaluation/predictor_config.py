"""Identity of the fitted predictor, separate from its probability calibration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PredictorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["final", "cv_mean"] = "final"
    weighting: Literal["equal"] = "equal"
