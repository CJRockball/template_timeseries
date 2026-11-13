"""Validated keyed continuous Tree SHAP in estimator-input space, not log-odds."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionTolerance:
    prediction_atol: float = 1e-5
    prediction_rtol: float = 1e-5

    def __post_init__(self):
        for value in vars(self).values():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("finite nonnegative explanation tolerances required")


@dataclass(frozen=True)
class RegressionExplanation:
    features: pd.DataFrame
    contributions: pd.DataFrame
    rows: pd.DataFrame
    verification: dict
    method: str = "xgboost_native_tree_shap"
    output_scale: str = "prediction_target_units"
    feature_dependence: str = "tree_path_dependent"


def build_regression_explanation(
    features, contributions, base_values, predictions, *, tolerance=None
):
    tol = RegressionTolerance() if tolerance is None else tolerance
    if not isinstance(tol, RegressionTolerance):
        raise TypeError("RegressionTolerance required")
    if (
        not isinstance(features, pd.DataFrame)
        or features.empty
        or not features.columns.is_unique
        or isinstance(features.index, pd.MultiIndex)
        or features.index.hasnans
        or not features.index.is_unique
        or not all(isinstance(c, str) and c for c in features.columns)
    ):
        raise ValueError("nonempty uniquely keyed features and unique names required")
    if isinstance(contributions, pd.DataFrame) and (
        not contributions.index.equals(features.index)
        or not contributions.columns.equals(features.columns)
    ):
        raise ValueError("contribution labels differ from features")
    for value in (base_values, predictions):
        if isinstance(value, pd.Series) and not value.index.equals(features.index):
            raise ValueError("prediction/baseline labels differ from features")
    values = np.asarray(contributions)
    base, predicted = np.asarray(base_values), np.asarray(predictions)
    n, width = features.shape
    if (
        values.shape != (n, width)
        or base.shape != (n,)
        or predicted.shape != (n,)
        or any(v.dtype.kind not in "iuf" for v in (values, base, predicted))
    ):
        raise ValueError("real numeric row-by-feature contributions and row vectors required")
    values, base, predicted = (v.astype(float) for v in (values, base, predicted))
    if not all(np.isfinite(v).all() for v in (values, base, predicted)):
        raise ValueError("nonfinite regression explanation")
    reconstructed = base + values.sum(axis=1)
    if not np.allclose(
        reconstructed, predicted, atol=tol.prediction_atol, rtol=tol.prediction_rtol
    ):
        raise ValueError("contributions do not reconstruct regression predictions")
    rows = pd.DataFrame(
        {
            "base_value": base,
            "y_pred_raw": predicted,
            "y_pred_reconstructed": reconstructed,
            "additivity_residual": reconstructed - predicted,
        },
        index=features.index.copy(),
    )
    return RegressionExplanation(
        features.copy(deep=True),
        pd.DataFrame(values.copy(), index=features.index.copy(), columns=features.columns.copy()),
        rows,
        {
            "passed": True,
            "n_rows": n,
            "n_features": width,
            "max_abs_additivity_residual": float(np.max(abs(reconstructed - predicted))),
            "tolerance": vars(tol).copy(),
            "labels_loaded": False,
        },
    )


def global_summary(result):
    values = result.contributions
    return (
        pd.DataFrame(
            {
                "feature": values.columns,
                "mean_abs_contribution": values.abs().mean().to_numpy(),
                "mean_contribution": values.mean().to_numpy(),
                "positive_fraction": values.gt(0).mean().to_numpy(),
                "n_rows": len(values),
                "output_scale": result.output_scale,
            }
        )
        .sort_values(["mean_abs_contribution", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
