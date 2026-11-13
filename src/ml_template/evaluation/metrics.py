"""Pure metric functions for binary classification.

No database, config or file access. Called by training (CV predictions) and by
evaluation (holdout, reruns on stored predictions).

Standard set: every run computes ``STANDARD_METRICS`` at ``STANDARD_ALERT_RATES``.
A config can add extras but cannot remove these, so any two runs are comparable.

Alert-rate metrics flag the top ``ceil(rate * n)`` rows by score. Ties are broken
by row order (stable sort), so results are deterministic.

Result keys: ``pr_auc``, ``roc_auc``, ``brier``, and for alert metrics
``recall_at_5pct`` / ``precision_at_5pct`` (one key per alert rate).
Undefined results (for example one class only) are NaN. Store them as NULL.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

NAN = float("nan")

METRIC_NAMES = frozenset(
    {"pr_auc", "roc_auc", "brier", "recall_at_alert_rate", "precision_at_alert_rate"}
)
ALERT_METRICS = frozenset({"recall_at_alert_rate", "precision_at_alert_rate"})

STANDARD_METRICS = (
    "pr_auc",
    "roc_auc",
    "brier",
    "recall_at_alert_rate",
    "precision_at_alert_rate",
)
STANDARD_ALERT_RATES = (0.05, 0.10)


def with_standard(
    names: Sequence[str] | None, alert_rates: Sequence[float] | None
) -> tuple[list[str], list[float]]:
    """Standard metrics and rates first, then any extras. Duplicates removed."""
    merged_names = list(STANDARD_METRICS) + [n for n in (names or []) if n not in STANDARD_METRICS]
    merged_rates = sorted({*STANDARD_ALERT_RATES, *(alert_rates or [])})
    return merged_names, merged_rates


def _prepare(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("y_true and y_score must be 1-D and of equal length")
    if len(y) == 0:
        raise ValueError("empty input")
    if not np.isfinite(s).all():
        raise ValueError("y_score contains NaN or inf")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    return y.astype(int), s


def _has_both_classes(y: np.ndarray) -> bool:
    return 0 < int(y.sum()) < len(y)


def _check_rate(rate: float) -> None:
    if not 0 < rate <= 1:
        raise ValueError(f"alert rate must be in (0, 1], got {rate}")


def alert_label(rate: float) -> str:
    """Stable name fragment for an alert rate, e.g. 0.05 -> '5pct'."""
    return f"{round(rate * 100, 6):g}pct"


def metric_key(name: str, rate: float | None = None) -> str:
    """Key used in result dicts and in the run_metrics table."""
    if name in ALERT_METRICS:
        if rate is None:
            raise ValueError(f"{name} needs an alert rate")
        _check_rate(rate)
        return f"{name.replace('_alert_rate', '')}_{alert_label(rate)}"
    return name


def resolved_keys(names: Sequence[str], alert_rates: Sequence[float] = ()) -> list[str]:
    """All result keys that ``compute_metrics`` will produce for this request."""
    unknown = set(names) - METRIC_NAMES
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; allowed: {sorted(METRIC_NAMES)}")
    keys: list[str] = []
    for name in names:
        if name in ALERT_METRICS:
            if not alert_rates:
                raise ValueError(f"{name} requested but alert_rates is empty")
            keys.extend(metric_key(name, rate) for rate in alert_rates)
        else:
            keys.append(name)
    return keys


def pr_auc(y_true: Any, y_score: Any) -> float:
    """Average precision. NaN if only one class is present."""
    y, s = _prepare(y_true, y_score)
    return float(average_precision_score(y, s)) if _has_both_classes(y) else NAN


def roc_auc(y_true: Any, y_score: Any) -> float:
    y, s = _prepare(y_true, y_score)
    return float(roc_auc_score(y, s)) if _has_both_classes(y) else NAN


def brier(y_true: Any, y_score: Any) -> float:
    """Brier score. ``y_score`` must be a probability in [0, 1]."""
    y, s = _prepare(y_true, y_score)
    if s.min() < 0 or s.max() > 1:
        raise ValueError("brier needs probabilities in [0, 1]")
    return float(brier_score_loss(y, s))


def _top_k_hits(y: np.ndarray, s: np.ndarray, rate: float) -> tuple[int, int]:
    _check_rate(rate)
    k = max(1, math.ceil(round(rate * len(y), 9)))
    order = np.argsort(-s, kind="stable")
    return int(y[order[:k]].sum()), k


def recall_at_alert_rate(y_true: Any, y_score: Any, rate: float) -> float:
    """Share of all positives found in the top ``rate`` of scores."""
    y, s = _prepare(y_true, y_score)
    positives = int(y.sum())
    hits, _ = _top_k_hits(y, s, rate)
    return hits / positives if positives else NAN


def precision_at_alert_rate(y_true: Any, y_score: Any, rate: float) -> float:
    """Share of the top ``rate`` of scores that are positive."""
    y, s = _prepare(y_true, y_score)
    hits, k = _top_k_hits(y, s, rate)
    return hits / k


def compute_metrics(
    y_true: Any,
    y_score: Any,
    names: Sequence[str],
    alert_rates: Sequence[float] = (),
) -> dict[str, float]:
    """Compute the requested metrics. Alert metrics expand once per rate."""
    resolved_keys(names, alert_rates)  # validates names and rates up front
    out: dict[str, float] = {}
    for name in names:
        if name == "recall_at_alert_rate":
            for rate in alert_rates:
                out[metric_key(name, rate)] = recall_at_alert_rate(y_true, y_score, rate)
        elif name == "precision_at_alert_rate":
            for rate in alert_rates:
                out[metric_key(name, rate)] = precision_at_alert_rate(y_true, y_score, rate)
        elif name == "pr_auc":
            out[name] = pr_auc(y_true, y_score)
        elif name == "roc_auc":
            out[name] = roc_auc(y_true, y_score)
        else:
            out[name] = brier(y_true, y_score)
    return out


def long_form_metrics(
    y_true: Any,
    y_score: Any,
    names: Sequence[str],
    alert_rates: Sequence[float] = (),
    *,
    split: str,
    folds: Any = None,
) -> list[dict[str, Any]]:
    """Rows shaped for the run_metrics table: split, fold, metric, value.

    ``fold = -1`` marks the pooled result over all rows. With ``folds`` given (one
    fold id per row), per-fold rows are added as well.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    rows: list[dict[str, Any]] = []

    def add(fold: int, yy: np.ndarray, ss: np.ndarray) -> None:
        for metric, value in compute_metrics(yy, ss, names, alert_rates).items():
            rows.append({"split": split, "fold": fold, "metric": metric, "value": value})

    add(-1, y, s)
    if folds is not None:
        f = np.asarray(folds)
        if len(f) != len(y):
            raise ValueError("folds must have one entry per row")
        for fold in sorted({int(x) for x in f}):
            mask = f == fold
            add(fold, y[mask], s[mask])
    return rows


def label_summary_rows(y_true: Any, *, split: str) -> list[dict[str, Any]]:
    """Row count, positive count and base rate, so PR-AUC can be read as lift over random."""
    y = np.asarray(y_true).astype(int)
    n, positives = len(y), int(y.sum())
    values = {"n_rows": float(n), "n_positives": float(positives), "base_rate": positives / n}
    return [{"split": split, "fold": -1, "metric": k, "value": v} for k, v in values.items()]


def fold_stability_rows(
    rows: Sequence[dict[str, Any]], keys: Sequence[str], *, split: str = "cv"
) -> list[dict[str, Any]]:
    """Mean and sample std of each metric across folds, stored at fold -1.

    Names: ``<metric>__fold_mean`` and ``<metric>__fold_std``. Needs 2+ valid folds.
    """
    out: list[dict[str, Any]] = []
    for key in keys:
        values = [
            r["value"]
            for r in rows
            if r["split"] == split
            and r["fold"] >= 0
            and r["metric"] == key
            and not math.isnan(r["value"])
        ]
        if len(values) < 2:
            continue
        out.append(
            {
                "split": split,
                "fold": -1,
                "metric": f"{key}__fold_mean",
                "value": float(np.mean(values)),
            }
        )
        out.append(
            {
                "split": split,
                "fold": -1,
                "metric": f"{key}__fold_std",
                "value": float(np.std(values, ddof=1)),
            }
        )
    return out
