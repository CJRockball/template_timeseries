"""Exploratory calibration diagnostics on stored OOF score-label pairs.

Cross-fitting here holds out labels from the calibrator, not from the complete
base-model selection process. This is not nested-CV or final-test evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ml_template.models.probability_calibrator import ProbabilityCalibrator, labels, probabilities


def probability_metrics(y, p) -> dict[str, float | int | None]:
    p = probabilities(p)
    y = labels(y, len(p))
    both = len(np.unique(y)) == 2
    return {
        "n_rows": len(y),
        "n_positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
    }


def reliability_table(y, p, n_bins: int = 5) -> pd.DataFrame:
    p = probabilities(p)
    y = labels(y, len(p))
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    groups = (
        np.zeros(len(p), dtype=int)
        if len(edges) == 1
        else np.searchsorted(edges[1:-1], p, side="right")
    )
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        n = int(mask.sum())
        positives = int(y[mask].sum())
        rate = positives / n
        denom = 1 + 1.96**2 / n
        centre = (rate + 1.96**2 / (2 * n)) / denom
        half = 1.96 * np.sqrt(rate * (1 - rate) / n + 1.96**2 / (4 * n**2)) / denom
        rows.append(
            {
                "bin": int(group),
                "n_rows": n,
                "n_positives": positives,
                "p_min": float(p[mask].min()),
                "p_max": float(p[mask].max()),
                "mean_probability": float(p[mask].mean()),
                "event_rate": rate,
                "event_rate_low": max(0.0, centre - half),
                "event_rate_high": min(1.0, centre + half),
            }
        )
    return pd.DataFrame(rows)


def cross_fitted_probabilities(
    p, y, folds, *, method="sigmoid", epsilon=1e-7, l2=1e-6, min_class_count=10
) -> np.ndarray:
    p = probabilities(p)
    y = labels(y, len(p))
    folds = np.asarray(folds)
    if folds.ndim != 1 or len(folds) != len(p) or pd.isna(folds).any():
        raise ValueError("folds must be aligned and non-null")
    if len(np.unique(folds)) < 2:
        raise ValueError("need at least two calibration diagnostic folds")
    output = np.empty(len(p))
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        counts = np.bincount(y[train].astype(int), minlength=2)
        if counts.min() < min_class_count:
            raise ValueError(f"calibration diagnostic training slice {fold} has too few of a class")
        calibrator = ProbabilityCalibrator.fit(
            p[train], y[train], method=method, epsilon=epsilon, l2=l2
        )
        output[test] = calibrator.transform(p[test])
    return output
