"""LightGBM binary classifier behind the model-adapter interface.

Needs lightgbm >= 4.0. Pandas category columns are used natively. Rounds are 1-based:
``best_round`` is the number of trees to keep. Saving and loading go through the native
booster, in LightGBM's text format, to whatever path is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ml_template.models.class_weight import resolve_scale_pos_weight
from ml_template.training.run_config import LGBMParams

CURVE_METRICS = {
    "logloss": "binary_logloss",
    "aucpr": "average_precision",
    "auc": "auc",
    "error": "binary_error",
}


def _supports_eval_xy() -> bool:
    """LightGBM 4.7 added eval_X and eval_y and deprecated eval_set."""
    major, minor, *_ = lgb.__version__.split(".")
    return (int(major), int(minor)) >= (4, 7)


_NEW_EVAL_ARGS = _supports_eval_xy()


def eval_args(sets: Sequence[tuple[pd.DataFrame, pd.Series]]) -> dict[str, object]:
    """Validation arguments for fit(), in the form the installed LightGBM expects."""
    if _NEW_EVAL_ARGS:
        return {"eval_X": tuple(X for X, _ in sets), "eval_y": tuple(y for _, y in sets)}
    return {"eval_set": list(sets)}


def curve_rows(
    history: Mapping[str, Mapping[str, Sequence[float]]], native_metric: str, label: str
) -> pd.DataFrame:
    """Long table (round, dataset, metric, value) from LightGBM's evals_result_."""
    parts = []
    for dataset in ("train", "val"):
        values = np.asarray(history[dataset][native_metric], dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "round": np.arange(1, len(values) + 1),
                    "dataset": dataset,
                    "metric": label,
                    "value": values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


class LGBMAdapter:
    def __init__(self, params: LGBMParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.max_rounds = params.n_estimators
        self.model: LGBMClassifier | None = None  # set after fitting
        self._loaded_booster: lgb.Booster | None = None  # set after load()

    def _build(self, y: pd.Series, n_estimators: int, metric: str | None = None) -> LGBMClassifier:
        p = self.params
        return LGBMClassifier(
            objective=p.objective,
            metric=metric or p.eval_metric,
            n_estimators=n_estimators,
            num_leaves=p.num_leaves,
            max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            subsample=p.subsample,
            subsample_freq=p.subsample_freq,
            colsample_bytree=p.colsample_bytree,
            min_child_samples=p.min_child_samples,
            reg_lambda=p.reg_lambda,
            reg_alpha=p.reg_alpha,
            max_bin=p.max_bin,
            scale_pos_weight=resolve_scale_pos_weight(p.scale_pos_weight, y),
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbosity=-1,
        )

    def _booster(self) -> lgb.Booster:
        if self.model is not None:
            return self.model.booster_
        if self._loaded_booster is not None:
            return self._loaded_booster
        raise RuntimeError("model is not fitted")

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int:
        """Fit with early stopping on the validation slice. Returns the best round count."""
        model = self._build(y_train, self.max_rounds)
        stopper = lgb.early_stopping(
            self.params.early_stopping_rounds, first_metric_only=True, verbose=False
        )
        model.fit(X_train, y_train, callbacks=[stopper], **eval_args([(X_val, y_val)]))
        self.model, self._loaded_booster = model, None
        return int(model.best_iteration_ or self.max_rounds)

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None:
        """Fit a fixed number of rounds with no early stopping."""
        model = self._build(y, int(n_rounds))
        model.fit(X, y)
        self.model, self._loaded_booster = model, None

    def learning_curve(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_rounds: int,
        metric: str,
    ) -> pd.DataFrame:
        """Metric per round on the training and validation rows. No early stopping."""
        native = CURVE_METRICS[metric]
        model = self._build(y_train, int(n_rounds), metric=native)
        model.fit(
            X_train,
            y_train,
            eval_names=["train", "val"],
            **eval_args([(X_train, y_train), (X_val, y_val)]),
        )
        return curve_rows(model.evals_result_, native, metric)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class."""
        if self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        return np.asarray(self._booster().predict(X), dtype=float)

    def feature_importances(self) -> dict[str, float]:
        """Total gain per feature. Features the model never split on get 0."""
        booster = self._booster()
        gains = booster.feature_importance(importance_type="gain")
        return {name: float(g) for name, g in zip(booster.feature_name(), gains, strict=True)}

    def save(self, path: Path) -> None:
        self._booster().save_model(str(path))

    @classmethod
    def load(cls, params: LGBMParams, seed: int, path: Path) -> LGBMAdapter:
        adapter = cls(params, seed)
        adapter._loaded_booster = lgb.Booster(model_file=str(path))
        return adapter
