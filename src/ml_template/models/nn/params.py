"""Torch-free NN schema for the current binary production pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NNPreprocessingParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    numeric: tuple[str, ...] | None = None
    categorical: tuple[str, ...] | None = None
    numeric_strategy: Literal["mean", "median", "constant"] = "median"
    scaling: Literal["standard", "none"] = "standard"
    constant_fill: float = 0.0
    all_missing_fill: float = 0.0
    unknown_policy: Literal["reserved", "error"] = "reserved"

    @model_validator(mode="after")
    def check_columns(self):
        for values in (self.numeric, self.categorical):
            if values is not None and (
                len(set(values)) != len(values) or any(not c for c in values)
            ):
                raise ValueError("feature lists require unique nonempty names")
        if self.numeric is not None and self.categorical is not None:
            if not self.numeric and not self.categorical:
                raise ValueError("both source input branches cannot be empty")
            if set(self.numeric) & set(self.categorical):
                raise ValueError("numeric/categorical inputs overlap")
        return self


class NNParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    architecture: Literal["mlp"] = "mlp"
    task: Literal["binary_classification"] = "binary_classification"
    preprocessing: NNPreprocessingParams = Field(default_factory=NNPreprocessingParams)
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = Field(0.1, ge=0, lt=1)
    embedding_dropout: float = Field(0.1, ge=0, lt=1)
    embedding_max_dim: int = Field(20, ge=1)
    numeric_projection_multiplier: int = Field(4, ge=1)
    normalization: Literal["layernorm", "none"] = "layernorm"
    max_epochs: int = Field(100, ge=1)
    patience: int = Field(5, ge=1)
    min_delta: float = Field(0.0, ge=0)
    monitor: Literal["logloss", "auc", "aucpr"] = "logloss"
    batch_size: int = Field(1024, ge=1)
    prediction_batch_size: int = Field(4096, ge=1)
    learning_rate: float = Field(0.0001, gt=0)
    weight_decay: float = Field(0.000001, ge=0)
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: Literal["none", "cosine"] = "none"
    gradient_clip_norm: float | None = Field(None, gt=0)
    pos_weight: float | Literal["auto"] = 1.0
    device: Literal["cpu", "cuda", "cuda:0"] = "cpu"
    inference_device: Literal["cpu", "cuda", "cuda:0"] = "cpu"
    deterministic: bool = True
    cpu_threads: int = Field(1, ge=1)

    @model_validator(mode="after")
    def check_network(self):
        if not self.hidden_dims or any(n < 1 for n in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.pos_weight != "auto" and self.pos_weight <= 0:
            raise ValueError("pos_weight must be positive or auto")
        return self


class NNRegressionParams(NNParams):
    """Continuous-output MLP. Binary weighting is fixed and never used."""

    task: Literal["regression"] = "regression"
    loss: Literal["mse"] = "mse"
    monitor: Literal["rmse", "mae"] = "rmse"
    pos_weight: Literal[1.0] = 1.0


class NNModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["nn"] = "nn"
    params: NNParams | NNRegressionParams = Field(default_factory=NNParams)
