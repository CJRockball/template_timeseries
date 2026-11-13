"""Pure finite-vector regression metrics; no probability interpretation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_DIRECTIONS = {"rmse": "min", "mae": "min", "r2": "max"}


def regression_metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Targets and predictions must be aligned nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Targets and predictions must be finite")
    result = {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }
    if len(y) >= 2 and np.any(y != y[0]):
        result["r2"] = float(r2_score(y, p))
    return result


def regression_metric_rows(y, prediction, folds):
    y, p, f = np.asarray(y), np.asarray(prediction), np.asarray(folds)
    if f.shape != y.shape:
        raise ValueError("Fold labels must align with targets")
    rows = []
    for fold in [-1, *sorted(int(k) for k in np.unique(f))]:
        mask = np.ones(len(y), dtype=bool) if fold == -1 else f == fold
        rows.extend(
            {"split": "cv", "fold": fold, "metric": name, "value": value}
            for name, value in regression_metrics(y[mask], p[mask]).items()
        )
    return rows
