"""Cross-validation on the stored split, with early stopping.

For each fold, early stopping runs on an inner slice of that fold's training rows
and never sees the validation fold. The model is then refitted on all of the
fold's training rows for the chosen number of rounds and predicts the validation
fold. This is the same procedure used for the final model, so the CV score
estimates what the final model will do. The holdout is never touched here.

The refitted fold models are returned with the result, so they can be saved and
analysed (for example permutation importance) on rows they never saw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_template.evaluation.metrics import (
    fold_stability_rows,
    label_summary_rows,
    long_form_metrics,
    resolved_keys,
)
from ml_template.models.model_factory import AdapterFactory, ModelAdapter

logger = logging.getLogger("ml_template.training.cross_validation")


@dataclass(frozen=True)
class CVResult:
    oof: pd.DataFrame  # key, fold, p_oof
    folds: pd.DataFrame  # one row per fold: sizes, best_round, hit_cap
    metric_rows: list[dict[str, Any]]  # shaped for run_metrics
    n_rounds_final: int
    fold_models: dict[int, ModelAdapter]  # the model that predicted each fold
    fold_val_keys: dict[int, np.ndarray]  # that fold's validation keys

    early_stopping_curves: pd.DataFrame | None = None


def _check_alignment(
    X: pd.DataFrame, y: pd.Series, assignments: pd.DataFrame, key_col: str
) -> None:
    if not X.index.is_unique:
        raise ValueError("X index (entity keys) must be unique")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if set(assignments[key_col]) != set(X.index):
        raise ValueError("split assignments and data cover different rows")


def inner_split(
    train_keys: np.ndarray,
    y: pd.Series,
    fraction: float,
    seed: int,
    task: str = "binary_classification",
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of a fold's training keys into fit and early-stopping slices."""
    fit_keys, stop_keys = train_test_split(
        train_keys,
        test_size=fraction,
        stratify=y.loc[train_keys].to_numpy() if task == "binary_classification" else None,
        random_state=seed,
    )
    return fit_keys, stop_keys


def cross_validate(
    make_adapter: AdapterFactory,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    key_col: str,
    *,
    validation_fraction: float,
    seed: int,
    metric_names: list[str],
    alert_rates: list[float],
    refit_rounds_factor: float = 1.0,
    task: str = "binary_classification",
) -> CVResult:
    history_parts = []
    if task not in {"binary_classification", "regression"}:
        raise ValueError(f"Unsupported task: {task!r}")
    prediction_col = "y_pred_oof" if task == "regression" else "p_oof"
    _check_alignment(X, y, assignments, key_col)
    dev = assignments[assignments["partition"] == "dev"]
    fold_ids = sorted(int(k) for k in dev["fold"].unique())
    if len(fold_ids) < 2:
        raise ValueError("need at least 2 CV folds")
    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_models: dict[int, ModelAdapter] = {}
    fold_val_keys: dict[int, np.ndarray] = {}
    for k in fold_ids:
        val_keys = dev.loc[dev["fold"] == k, key_col].to_numpy()
        train_keys = dev.loc[dev["fold"] != k, key_col].to_numpy()
        if task == "regression":
            fit_keys, stop_keys = inner_split(train_keys, y, validation_fraction, seed, task=task)
        else:
            fit_keys, stop_keys = inner_split(train_keys, y, validation_fraction, seed)
        if task == "binary_classification" and (
            y.loc[stop_keys].sum() == 0 or y.loc[fit_keys].sum() == 0
        ):
            raise ValueError(f"fold {k}: inner split has a slice without positives")
        stopper: ModelAdapter = make_adapter()
        best_round = stopper.fit_with_validation(
            X.loc[fit_keys], y.loc[fit_keys], X.loc[stop_keys], y.loc[stop_keys]
        )
        from ml_template.evaluation.curve_history import capture_inner_history

        captured = capture_inner_history(stopper, best_round, k)
        if captured is not None:
            history_parts.append(captured)
        hit_cap = best_round >= stopper.max_rounds
        if hit_cap:
            logger.warning(
                "fold %d: selected round/epoch reached the cap (%d); consider increasing the model training cap",
                k,
                best_round,
            )
        fitter: ModelAdapter = make_adapter()
        fitter.refit(X.loc[train_keys], y.loc[train_keys], best_round)
        p = (
            fitter.predict(X.loc[val_keys])
            if task == "regression"
            else fitter.predict_proba(X.loc[val_keys])
        )
        p = np.asarray(p)
        if p.shape != (len(val_keys),) or not np.isfinite(p).all():
            raise ValueError("Invalid validation predictions")
        oof_parts.append(pd.DataFrame({key_col: val_keys, "fold": k, prediction_col: p}))
        fold_models[k] = fitter
        fold_val_keys[k] = val_keys
        fold_rows.append(
            {
                "fold": k,
                "n_train": len(train_keys),
                "n_inner_fit": len(fit_keys),
                "n_inner_stop": len(stop_keys),
                "n_val": len(val_keys),
                "best_round": best_round,
                "hit_cap": hit_cap,
            }
        )
        logger.info("fold %d: best_round=%d, n_val=%d", k, best_round, len(val_keys))
    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    y_oof = y.loc[oof[key_col]].to_numpy()
    if task == "regression":
        from ml_template.evaluation.regression_metrics import regression_metric_rows

        rows = regression_metric_rows(y_oof, oof[prediction_col].to_numpy(), oof["fold"].to_numpy())
        rows += fold_stability_rows(rows, ["rmse", "mae", "r2"])
    else:
        rows = long_form_metrics(
            y_oof,
            oof["p_oof"].to_numpy(),
            metric_names,
            alert_rates,
            split="cv",
            folds=oof["fold"].to_numpy(),
        )
        rows += fold_stability_rows(rows, resolved_keys(metric_names, alert_rates))
        rows += label_summary_rows(y_oof, split="cv")
    n_final = max(1, round(float(np.median(folds["best_round"])) * refit_rounds_factor))
    rows += [
        {"split": "cv", "fold": int(r.fold), "metric": "best_round", "value": float(r.best_round)}
        for r in folds.itertuples()
    ]
    rows.append({"split": "cv", "fold": -1, "metric": "n_rounds_final", "value": float(n_final)})
    return CVResult(
        oof=oof,
        folds=folds,
        metric_rows=rows,
        n_rounds_final=n_final,
        fold_models=fold_models,
        fold_val_keys=fold_val_keys,
        early_stopping_curves=pd.concat(history_parts, ignore_index=True)
        if history_parts
        else None,
    )


def fit_final(
    make_adapter: AdapterFactory,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    key_col: str,
    n_rounds: int,
) -> ModelAdapter:
    """Fit on all development rows for a fixed number of rounds. Holdout rows are excluded."""
    _check_alignment(X, y, assignments, key_col)
    dev_keys = assignments.loc[assignments["partition"] == "dev", key_col].to_numpy()
    adapter = make_adapter()
    adapter.refit(X.loc[dev_keys], y.loc[dev_keys], n_rounds)
    return adapter
