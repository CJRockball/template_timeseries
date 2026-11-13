"""XGBoost binary classifier behind the model-adapter interface.

Needs xgboost >= 2.0 (``device`` argument, early stopping in the constructor).
Categorical columns use XGBoost's native support, which needs no fitted encoder.
Rounds are 1-based: ``best_round`` is the number of trees to keep.

Saving and loading go through the native booster, not the scikit-learn wrapper.
The wrapper's ``save_model`` reads ``_estimator_type``, which newer scikit-learn
releases no longer define, so it fails on some xgboost/scikit-learn pairings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from ml_template.training.run_config import XGBParams


def resolve_scale_pos_weight(value: float | str, y: pd.Series) -> float:
    """'auto' means negatives over positives in the rows being fitted."""
    if value == "auto":
        positives = int(y.sum())
        if positives == 0:
            raise ValueError("cannot compute scale_pos_weight: no positives in training rows")
        return (len(y) - positives) / positives
    return float(value)


def curve_rows(history: Mapping[str, Mapping[str, Sequence[float]]], metric: str) -> pd.DataFrame:
    """Long table (round, dataset, metric, value) from XGBoost's evals_result."""
    parts = []
    for key, label in (("validation_0", "train"), ("validation_1", "val")):
        values = np.asarray(history[key][metric], dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "round": np.arange(1, len(values) + 1),
                    "dataset": label,
                    "metric": metric,
                    "value": values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


class XGBAdapter:
    def __init__(self, params: XGBParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.max_rounds = params.n_estimators
        self.model: XGBClassifier | None = None  # set after fitting
        self._loaded_booster: xgb.Booster | None = None  # set after load()

    def _build(
        self, y: pd.Series, n_estimators: int, early_stopping_rounds: int | None
    ) -> XGBClassifier:
        p = self.params
        return XGBClassifier(
            objective=p.objective,
            eval_metric=p.eval_metric,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            subsample=p.subsample,
            colsample_bytree=p.colsample_bytree,
            min_child_weight=p.min_child_weight,
            reg_lambda=p.reg_lambda,
            reg_alpha=p.reg_alpha,
            max_bin=p.max_bin,
            scale_pos_weight=resolve_scale_pos_weight(p.scale_pos_weight, y),
            device=p.device,
            tree_method="hist",
            enable_categorical=True,
            random_state=self.seed,
            verbosity=0,
        )

    def _booster(self) -> xgb.Booster:
        if self.model is not None:
            return self.model.get_booster()
        if self._loaded_booster is not None:
            return self._loaded_booster
        raise RuntimeError("model is not fitted")

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int:
        """Fit with early stopping on the validation slice. Returns the best round count."""
        model = self._build(y_train, self.max_rounds, self.params.early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.model, self._loaded_booster = model, None
        return int(model.best_iteration) + 1

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None:
        """Fit a fixed number of rounds with no early stopping."""
        model = self._build(y, int(n_rounds), None)
        model.fit(X, y, verbose=False)
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
        model = self._build(y_train, int(n_rounds), None)
        model.set_params(eval_metric=metric)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        return curve_rows(model.evals_result(), metric)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class."""
        if self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        booster = self._booster()
        return booster.predict(xgb.DMatrix(X, enable_categorical=True))

    def estimator_feature_names(self) -> list[str]:
        """Ordered names seen by the fitted estimator, including fitted features."""
        names = self._booster().feature_names
        if not names:
            raise ValueError("explanations require saved estimator feature names")
        return list(names)

    def _margin_prediction_inputs(self, X: pd.DataFrame):
        import json

        booster = self._booster()
        objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
        if objective != "binary:logistic":
            raise ValueError("margin explanations support only binary:logistic")
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("nonempty feature DataFrame required")
        if list(X.columns) != self.estimator_feature_names():
            raise ValueError("feature order differs from estimator")
        iteration_range = (0, 0)
        if self.model is not None:
            try:
                best_iteration = self.model.best_iteration
            except AttributeError:
                pass
            else:
                iteration_range = (0, int(best_iteration) + 1)
        return booster, xgb.DMatrix(X, enable_categorical=True), iteration_range

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        """Raw log-odds with the same tree range as predict_proba."""
        booster, matrix, rounds = self._margin_prediction_inputs(X)
        margin = np.asarray(
            booster.predict(matrix, output_margin=True, strict_shape=True, iteration_range=rounds)
        )
        if margin.shape != (len(X), 1):
            raise ValueError("unexpected binary margin prediction shape")
        return margin[:, 0]

    def explain_margin(self, X: pd.DataFrame, *, tolerance=None):
        """Exact native Tree SHAP in log-odds; never calibrated-probability attribution."""
        from ml_template.explainability.result import build_margin_explanation

        booster, matrix, rounds = self._margin_prediction_inputs(X)
        native = np.asarray(
            booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=False,
                strict_shape=True,
                iteration_range=rounds,
            )
        )
        if native.shape != (len(X), 1, len(X.columns) + 1):
            raise ValueError("unexpected binary contribution prediction shape")
        margin = np.asarray(
            booster.predict(matrix, output_margin=True, strict_shape=True, iteration_range=rounds)
        )
        if margin.shape != (len(X), 1):
            raise ValueError("unexpected binary margin prediction shape")
        return build_margin_explanation(
            X,
            native[:, 0, :-1],
            native[:, 0, -1],
            margin[:, 0],
            self.predict_proba(X),
            tolerance=tolerance,
        )

    def feature_importances(self) -> dict[str, float]:
        """Total gain per feature. Features the model never split on get 0."""
        booster = self._booster()
        scores = booster.get_score(importance_type="gain")
        return {name: float(scores.get(name, 0.0)) for name in booster.feature_names}

    def save(self, path: Path) -> None:
        self._booster().save_model(str(path))

    @classmethod
    def load(cls, params: XGBParams, seed: int, path: Path) -> XGBAdapter:
        adapter = cls(params, seed)
        booster = xgb.Booster()
        booster.load_model(str(path))
        adapter._loaded_booster = booster
        return adapter
