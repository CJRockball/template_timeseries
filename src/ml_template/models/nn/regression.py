"""Continuous NN targets, metrics and minimum-direction early stopping (torch-free)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def targets(X, y):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("training features must be a nonempty DataFrame with unique columns")
    if not X.index.is_unique or X.index.hasnans:
        raise ValueError("training keys must be unique and nonmissing")
    if not isinstance(y, pd.Series) or not X.index.equals(y.index):
        raise ValueError("targets must be a Series aligned with feature rows")
    if (
        not pd.api.types.is_numeric_dtype(y.dtype)
        or pd.api.types.is_bool_dtype(y.dtype)
        or pd.api.types.is_complex_dtype(y.dtype)
    ):
        raise ValueError("regression targets must be real numeric, not booleans or strings")
    values = y.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("regression targets must be finite and nonmissing")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("regression targets are outside float32 range")
    return result


def metric_value(name, truth, prediction):
    if name not in {"mse", "rmse", "mae"}:
        raise ValueError("unsupported NN regression metric")
    vectors = [np.asarray(v) for v in (truth, prediction)]
    if any(v.ndim != 1 or v.dtype.kind not in "iuf" for v in vectors):
        raise ValueError("expected real numeric vectors")
    if vectors[0].shape != vectors[1].shape or not len(vectors[0]):
        raise ValueError("metric requires nonempty aligned vectors")
    if any(not np.isfinite(v).all() for v in vectors):
        raise ValueError("nonfinite regression metric input")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = vectors[0].astype(np.float64) - vectors[1].astype(np.float64)
        value = np.mean(np.abs(residual)) if name == "mae" else np.mean(residual**2)
        if name == "rmse":
            value = np.sqrt(value)
    if not np.isfinite(value):
        raise ValueError("regression metric overflow")
    return float(value)


@dataclass
class RegressionStop:
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")

    def update(self, value, epoch):
        if not np.isfinite(value) or type(epoch) is not int or epoch < 1:
            raise ValueError("invalid stopping observation")
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = float(value), epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience
