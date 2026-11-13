"""Replacement-capable numeric preprocessing reusing the saved NN implementation.

Non-selected columns retain their existing representation, including categories.
This stage belongs AFTER fitted feature generation, inside each model fit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd

from ml_template.preprocessing.nn_config import NNPreprocessorConfig
from ml_template.preprocessing.nn_preprocessor import NNPreprocessor


class NumericPreprocessingStep:
    name = "numeric_preprocessing"
    replacement_stage = True

    def __init__(
        self,
        columns: Sequence[str],
        numeric_strategy="median",
        scaling="none",
        constant_fill=0.0,
        all_missing_fill=0.0,
        missing_indicators=True,
    ):
        if isinstance(columns, str) or not isinstance(columns, (list, tuple)):
            raise ValueError("columns must be an explicit list")
        if type(missing_indicators) is not bool:
            raise ValueError("missing_indicators must be boolean")
        self.config = NNPreprocessorConfig(
            numeric=tuple(columns),
            categorical=(),
            numeric_strategy=numeric_strategy,
            scaling=scaling,
            constant_fill=constant_fill,
            all_missing_fill=all_missing_fill,
        )
        self.missing_indicators = missing_indicators
        self.preprocessor = None
        self.input_columns = None

    def output_columns(self):
        return [f"{c}__is_missing" for c in self.config.numeric] if self.missing_indicators else []

    def _check(self, X, *, fitting=False):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("numeric preprocessing requires unique DataFrame columns")
        if not set(self.config.numeric) <= set(X.columns):
            raise ValueError("numeric preprocessing column missing")
        if fitting:
            if set(self.output_columns()) & set(X.columns):
                raise ValueError("missing-indicator output collides with an input")
        elif self.input_columns != list(X.columns):
            raise ValueError("numeric preprocessing input order/schema differs from fitted state")

    def fit(self, X):
        self.preprocessor, self.input_columns = None, None
        self._check(X, fitting=True)
        pre = NNPreprocessor(self.config).fit(X.loc[:, list(self.config.numeric)])
        self.preprocessor, self.input_columns = pre, list(X.columns)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise RuntimeError("numeric preprocessing is not fitted")
        self._check(X)
        values = self.preprocessor.transform(X.loc[:, list(self.config.numeric)])
        result = X.copy(deep=True)
        for name in self.config.numeric:
            result[name] = values[name]
        for name in self.output_columns():
            result[name] = values[name]
        return result


def validate_regression_specs(specs):
    """Do not accidentally enable binary-supervised target encoding for regression."""
    from ml_template.features.fitted_features import validate_specs

    if specs:
        validate_specs(specs)
        allowed = {"numeric_preprocessing", "power_transform", "group_statistics"}
        if any(spec.get("step") not in allowed for spec in specs):
            raise ValueError("Regression supports target-free fitted steps only")


def validate_saved_specs(transformer, specs):
    saved = json.dumps(transformer.specs, sort_keys=True, allow_nan=False)
    expected = json.dumps([dict(s) for s in specs], sort_keys=True, allow_nan=False)
    if saved != expected:
        raise ValueError("saved fitted transformer specification differs from run config")
