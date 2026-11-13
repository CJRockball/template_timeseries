"""CatBoost binary classifier behind the model-adapter interface.

Needs catboost >= 1.2. Non-numeric columns (category, object, string) are converted to text and
passed as categorical features; CatBoost encodes them itself, so no fitted encoder is needed.
Rounds are 1-based: ``best_round`` is the number of trees to keep. Models are saved in the
native ``cbm`` format to whatever path is given, and no files are written during training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ml_template.models.class_weight import resolve_scale_pos_weight
from ml_template.training.run_config import CatBoostParams

CURVE_METRICS = {
    "logloss": "Logloss",
    "aucpr": "PRAUC",
    "auc": "AUC",
    "error": "ZeroOneLoss",
}


def curve_rows(
    history: Mapping[str, Mapping[str, Sequence[float]]], native_metric: str, label: str
) -> pd.DataFrame:
    """Long table (round, dataset, metric, value) from CatBoost's evals result."""
    parts = []
    for key, dataset in (("validation_0", "train"), ("validation_1", "val")):
        values = np.asarray(history[key][native_metric], dtype=float)
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


def prepare(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Text-valued copies of non-numeric columns, and the names of those columns."""
    out = X.copy()
    cats = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    for col in cats:
        out[col] = out[col].astype(str)
    return out, cats


class CatBoostAdapter:
    def __init__(self, params: CatBoostParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.max_rounds = params.n_estimators
        self.model: CatBoostClassifier | None = None  # set after fitting or loading

    def _build(
        self, y: pd.Series, n_rounds: int, cats: list[str], eval_metric: str | None = None
    ) -> CatBoostClassifier:
        p = self.params
        return CatBoostClassifier(
            loss_function=p.loss_function,
            eval_metric=eval_metric or p.eval_metric,
            iterations=n_rounds,
            depth=p.depth,
            learning_rate=p.learning_rate,
            bootstrap_type="Bernoulli",
            subsample=p.subsample,
            rsm=p.rsm,
            l2_leaf_reg=p.l2_leaf_reg,
            border_count=p.border_count,
            scale_pos_weight=resolve_scale_pos_weight(p.scale_pos_weight, y),
            cat_features=cats or None,
            random_seed=self.seed,
            allow_writing_files=False,
            verbose=False,
        )

    def _fitted(self) -> CatBoostClassifier:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int:
        """Fit with early stopping on the validation slice. Returns the best round count."""
        Xt, cats = prepare(X_train)
        Xv, _ = prepare(X_val)
        model = self._build(y_train, self.max_rounds, cats)
        model.fit(
            Xt,
            y_train,
            eval_set=(Xv, y_val),
            early_stopping_rounds=self.params.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
        self.model = model
        return int(model.get_best_iteration()) + 1

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None:
        """Fit a fixed number of rounds with no early stopping."""
        Xp, cats = prepare(X)
        model = self._build(y, int(n_rounds), cats)
        model.fit(Xp, y, verbose=False)
        self.model = model

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
        Xt, cats = prepare(X_train)
        Xv, _ = prepare(X_val)
        model = self._build(y_train, int(n_rounds), cats, eval_metric=native)
        model.fit(
            Xt, y_train, eval_set=[(Xt, y_train), (Xv, y_val)], use_best_model=False, verbose=False
        )
        return curve_rows(model.get_evals_result(), native, metric)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class."""
        Xp, _ = prepare(X)
        return np.asarray(self._fitted().predict_proba(Xp)[:, 1], dtype=float)

    def feature_importances(self) -> dict[str, float]:
        """Prediction-values-change importance per feature."""
        model = self._fitted()
        scores = model.get_feature_importance()
        return {n: float(s) for n, s in zip(model.feature_names_, scores, strict=True)}

    def save(self, path: Path) -> None:
        self._fitted().save_model(str(path), format="cbm")

    @classmethod
    def load(cls, params: CatBoostParams, seed: int, path: Path) -> CatBoostAdapter:
        adapter = cls(params, seed)
        model = CatBoostClassifier()
        model.load_model(str(path), format="cbm")
        adapter.model = model
        return adapter
