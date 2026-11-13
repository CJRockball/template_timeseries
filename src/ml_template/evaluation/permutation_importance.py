"""Permutation importance on held-out rows, for any model with ``predict_proba``.

Each feature is shuffled in turn in a fold's validation rows, and the drop in
PR-AUC is the importance (positive = the model relies on it). Computed per fold
with that fold's own model, on rows it never saw. The pooled row holds the mean
across folds and, as ``std``, the spread across folds: a stable feature has a
large mean and a small spread.

Importance is a hint, not a verdict. With correlated features the drop is shared
and each looks weaker than the pair really is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_template.evaluation.metrics import roc_auc

METHOD = "permutation_roc_auc_drop"  # "permutation_pr_auc_drop"


def _subsample(
    X: pd.DataFrame, y: pd.Series, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(X) <= max_rows:
        return X, y
    keep, _ = train_test_split(
        np.arange(len(X)), train_size=max_rows, stratify=y.to_numpy(), random_state=seed
    )
    keep = np.sort(keep)
    return X.iloc[keep], y.iloc[keep]


def permutation_importance(
    adapter: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    seed: int = 0,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """One row per feature: mean and std (across repeats) of the PR-AUC drop."""
    if not X.index.is_unique:
        raise ValueError("X index must be unique")
    X, y = _subsample(X, y, max_rows, seed)
    y_arr = y.to_numpy()
    baseline = roc_auc(y_arr, adapter.predict_proba(X))  # pr_auc(y_arr, adapter.predict_proba(X))
    if math.isnan(baseline):
        raise ValueError("validation rows contain a single class; PR-AUC is undefined")

    rng = np.random.default_rng(seed)
    work = X.copy()
    rows = []
    for col in X.columns:
        original = X[col]
        drops = []
        for _ in range(n_repeats):
            shuffled = original.iloc[rng.permutation(len(X))]
            shuffled.index = X.index  # keeps the dtype, including categories
            work[col] = shuffled
            drops.append(baseline - roc_auc(y_arr, adapter.predict_proba(work)))
            # drops.append(baseline - pr_auc(y_arr, adapter.predict_proba(work)))
        work[col] = original
        rows.append(
            {
                "feature": col,
                "mean": float(np.mean(drops)),
                "std": float(np.std(drops, ddof=1)) if n_repeats > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def importance_rows_for_run(
    fold_models: Mapping[int, Any],
    fold_val_keys: Mapping[int, np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int,
    seed: int,
    max_rows: int | None,
    split: str = "cv",
) -> list[dict[str, Any]]:
    """Rows for the run_feature_importance table: per fold, plus pooled at fold -1."""
    rows: list[dict[str, Any]] = []
    per_fold: dict[int, pd.DataFrame] = {}
    for fold in sorted(fold_models):
        keys = fold_val_keys[fold]
        table = permutation_importance(
            fold_models[fold],
            X.loc[keys],
            y.loc[keys],
            n_repeats=n_repeats,
            seed=seed + int(fold),
            max_rows=max_rows,
        )
        per_fold[fold] = table.set_index("feature")
        for feature, rec in per_fold[fold].iterrows():
            rows.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "method": METHOD,
                    "feature": feature,
                    "value": float(rec["mean"]),
                    "std": float(rec["std"]),
                }
            )

    means = pd.DataFrame({f: t["mean"] for f, t in per_fold.items()})
    for feature in means.index:
        values = means.loc[feature].to_numpy(dtype=float)
        rows.append(
            {
                "split": split,
                "fold": -1,
                "method": METHOD,
                "feature": feature,
                "value": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            }
        )
    return rows
