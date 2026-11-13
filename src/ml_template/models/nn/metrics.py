"""Torch-independent binary monitoring and one-based early stopping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite 1-D vector")
    return expit(values)


def metric_value(name, y, p):
    if name == "logloss":
        return float(log_loss(y, p, labels=[0, 1]))
    if name == "auc":
        return float(roc_auc_score(y, p))
    if name == "aucpr":
        return float(average_precision_score(y, p))
    if name == "error":
        return float(np.mean((p >= 0.5) != y))
    raise ValueError(f"unknown metric: {name}")


@dataclass
class EarlyStop:
    patience: int
    metric: str
    min_delta: float = 0.0
    best_value: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self):
        if (
            self.patience < 1
            or self.metric not in ("logloss", "auc", "aucpr")
            or self.min_delta < 0
        ):
            raise ValueError("invalid early-stopping settings")

    def update(self, value, epoch):
        if not np.isfinite(value) or epoch < 1:
            raise ValueError("invalid validation metric/epoch")
        sign = -1 if self.metric == "logloss" else 1
        improved = self.best_value is None or sign * (value - self.best_value) > self.min_delta
        if improved:
            self.best_value, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience
