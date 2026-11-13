"""Build model adapters from a run config. The only place that knows model types.

Adding a neural network later means one new adapter module and one branch here.
Training and evaluation code depend only on the ``ModelAdapter`` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ml_template.training.run_config import RunConfig


class ModelAdapter(Protocol):
    max_rounds: int

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int: ...

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...


class RegressionAdapter(Protocol):
    max_rounds: int

    def fit_with_validation(self, X_train, y_train, X_val, y_val) -> int: ...
    def refit(self, X, y, n_rounds: int) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


AdapterFactory = Callable[[], ModelAdapter | RegressionAdapter]


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Give text columns the category dtype once, on the full frame.

    Done before any slicing so every fold shares one category mapping.
    """
    out = X.copy()
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[col] = series.astype("category")
    return out


def _adapter_class(model_type: str, task: str = "binary_classification") -> type:
    """The adapter class for a model type. Imports only the library that is needed."""
    if task == "regression":
        if model_type == "nn":
            from ml_template.models.nn_regressor import NNRegressionAdapter

            return NNRegressionAdapter
        if model_type != "xgb":
            raise ValueError("Only XGB and NN regression training are implemented")
        from ml_template.models.xgb_regressor import XGBRegressionAdapter

        return XGBRegressionAdapter
    if model_type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        return XGBAdapter
    if model_type == "lgbm":
        from ml_template.models.lgbm_classifier import LGBMAdapter

        return LGBMAdapter
    if model_type == "catboost":
        from ml_template.models.catboost_classifier import CatBoostAdapter

        return CatBoostAdapter
    if model_type == "nn":
        from ml_template.models.nn_classifier import NNAdapter

        return NNAdapter
    raise ValueError(f"unknown model type {model_type!r}")


def build_adapter_factory(cfg: RunConfig) -> AdapterFactory:
    """Return a callable that makes a fresh, unfitted adapter for this config."""
    adapter_class = _adapter_class(cfg.model.type, cfg.task)

    def base() -> ModelAdapter:
        return adapter_class(cfg.model.params, cfg.seed)

    if not cfg.fitted_features:
        return base
    from ml_template.features.fitted_features import TransformingAdapter

    return lambda: TransformingAdapter(base(), cfg.fitted_features)


def load_adapter(cfg: RunConfig, path: Path) -> ModelAdapter:
    inner = _adapter_class(cfg.model.type, cfg.task).load(cfg.model.params, cfg.seed, path)
    if not cfg.fitted_features:
        return inner
    from ml_template.features.fitted_features import TransformingAdapter

    return TransformingAdapter.load(inner, cfg.fitted_features, path)
