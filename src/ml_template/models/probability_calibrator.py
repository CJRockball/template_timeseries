"""Binary probability calibration; no model, database or file-system dependencies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def probabilities(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not len(p):
        raise ValueError("probabilities must be a nonempty 1-D array")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return p


def labels(values, n: int) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or len(y) != n or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be aligned binary 0/1 values")
    return y.astype(float)


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: Literal["none", "sigmoid"] = "sigmoid"
    epsilon: float = 1e-7
    l2: float = 1e-6
    slope: float = 1.0
    intercept: float = 0.0
    schema_version: int = 1
    score_transform: str = "clipped_logit"

    def __post_init__(self):
        if self.method not in {"none", "sigmoid"} or self.schema_version != 1:
            raise ValueError("unsupported calibrator schema or method")
        if self.score_transform != "clipped_logit":
            raise ValueError("unsupported score transform")
        if not np.isfinite([self.epsilon, self.l2, self.slope, self.intercept]).all():
            raise ValueError("calibrator parameters must be finite")
        if not 0 < self.epsilon < 0.01 or self.l2 < 0 or self.slope <= 0:
            raise ValueError("invalid epsilon, l2 or slope")

    @classmethod
    def fit(cls, p, y, *, method="sigmoid", epsilon=1e-7, l2=1e-6):
        base = cls(method=method, epsilon=epsilon, l2=l2)
        p = probabilities(p)
        y = labels(y, len(p))
        if len(np.unique(y)) != 2:
            raise ValueError("calibration requires both classes")
        if method == "none":
            return base
        z = logit(np.clip(p, epsilon, 1 - epsilon))

        def objective(theta):
            a, b = theta
            t = a * z + b
            residual = expit(t) - y
            loss = np.mean(np.logaddexp(0, t) - y * t) + 0.5 * l2 * a * a
            grad = np.array([np.mean(residual * z) + l2 * a, np.mean(residual)])
            return loss, grad

        initial_b = float(logit(y.mean()) - np.mean(z))
        result = minimize(
            objective,
            [1.0, initial_b],
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-8, None), (None, None)],
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"calibrator optimisation failed: {result.message}")
        return cls(
            method=method,
            epsilon=epsilon,
            l2=l2,
            slope=float(result.x[0]),
            intercept=float(result.x[1]),
        )

    def transform(self, p) -> np.ndarray:
        p = probabilities(p)
        if self.method == "none":
            return p.copy()
        z = logit(np.clip(p, self.epsilon, 1 - self.epsilon))
        return expit(self.slope * z + self.intercept)

    def save(self, path: Path):
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
