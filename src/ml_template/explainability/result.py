"""Validated binary-margin contributions in estimator feature space."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class ExplanationTolerance:
    margin_atol: float = 1e-5
    margin_rtol: float = 1e-5
    probability_atol: float = 1e-6
    probability_rtol: float = 1e-5

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("tolerances must be finite nonnegative numbers")
            if not np.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite nonnegative numbers")


@dataclass(frozen=True)
class MarginExplanation:
    """Owned copies of keyed tables; frozen fields do not make pandas objects immutable."""

    features: pd.DataFrame
    contributions: pd.DataFrame
    rows: pd.DataFrame
    verification: dict
    method: str = "xgboost_native_tree_shap"
    output_scale: str = "base_model_log_odds"
    feature_dependence: str = "tree_path_dependent"


class MarginExplainingAdapter(Protocol):
    """Optional capability, separate from the generic training ModelAdapter protocol."""

    def estimator_feature_names(self) -> list[str]: ...

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray: ...

    def explain_margin(
        self, X: pd.DataFrame, *, tolerance: ExplanationTolerance | None = None
    ) -> MarginExplanation: ...


def _vector(values, n, name):
    out = np.asarray(values, dtype=float)
    if out.shape != (n,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite vector with one value per row")
    return out


def build_margin_explanation(
    features, contributions, base_values, margins, p_raw, *, tolerance=None
):
    """Reject shape, key, additivity and probability-parity failures before returning."""
    tol = ExplanationTolerance() if tolerance is None else tolerance
    if not isinstance(tol, ExplanationTolerance):
        raise TypeError("tolerance must be ExplanationTolerance")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a nonempty DataFrame")
    if features.columns.duplicated().any() or not all(
        isinstance(c, str) and c for c in features.columns
    ):
        raise ValueError("feature names must be unique nonempty strings")
    if isinstance(features.index, pd.MultiIndex):
        raise ValueError("MultiIndex entity keys are unsupported")
    if features.index.has_duplicates or features.index.hasnans:
        raise ValueError("entity keys must be unique and non-null")
    n, width = features.shape
    if isinstance(contributions, pd.DataFrame) and (
        not contributions.index.equals(features.index)
        or not contributions.columns.equals(features.columns)
    ):
        raise ValueError("contribution labels differ from feature labels")
    for value in (base_values, margins, p_raw):
        if isinstance(value, pd.Series) and not value.index.equals(features.index):
            raise ValueError("row labels differ from feature labels")
    values = np.asarray(contributions, dtype=float)
    if values.shape != (n, width) or not np.isfinite(values).all():
        raise ValueError("contributions must be a finite row-by-feature matrix")
    base = _vector(base_values, n, "base_values")
    margin = _vector(margins, n, "margins")
    probability = _vector(p_raw, n, "p_raw")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("p_raw must be in [0, 1]")
    reconstructed = base + values.sum(axis=1)
    if not np.allclose(reconstructed, margin, atol=tol.margin_atol, rtol=tol.margin_rtol):
        raise ValueError("contributions do not reconstruct the base-model margin")
    from_margin = expit(margin)
    if not np.allclose(
        from_margin, probability, atol=tol.probability_atol, rtol=tol.probability_rtol
    ):
        raise ValueError("margin does not reproduce the base-model probability")
    rows = pd.DataFrame(
        {
            "base_value": base,
            "margin": margin,
            "p_raw": probability,
            "margin_reconstructed": reconstructed,
            "margin_residual": reconstructed - margin,
            "probability_residual": from_margin - probability,
        },
        index=features.index.copy(),
    )
    return MarginExplanation(
        features=features.copy(deep=True),
        contributions=pd.DataFrame(
            values.copy(), index=features.index.copy(), columns=features.columns
        ),
        rows=rows,
        verification={
            "passed": True,
            "n_rows": n,
            "n_features": width,
            "max_abs_margin_residual": float(np.max(np.abs(reconstructed - margin))),
            "max_abs_probability_residual": float(np.max(np.abs(from_margin - probability))),
            "tolerance": vars(tol).copy(),
            "labels_loaded": False,
        },
    )
