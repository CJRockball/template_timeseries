"""Feature steps that learn from data, and the model wrapper that runs them inside the folds.

Row-wise features are stored once as blocks. Features that learn statistics (group means,
power-transform parameters) must be fitted on training rows only, so they live here.
``TransformingAdapter`` wraps any model adapter: it fits the steps on exactly the rows its
inner model trains on, then applies them to whatever it predicts. Cross-validation, permutation
importance and holdout evaluation keep passing raw feature columns; nothing else changes.

Steps are listed under ``fitted_features`` in an experiment YAML::

    fitted_features:
      - {step: group_statistics, group_by: [type], columns: [torque_nm], stats: [mean, std]}
      - {step: power_transform, columns: [tool_wear_min], method: yeo-johnson}

Every step is fitted on the raw columns it receives and returns only new columns.
Supervised steps use fit_transform(X, y) for cross-fitted training values; inference
always uses transform(X) with the frozen full-training mappings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_template.features.blocks.generic_interactions import PowerTransformStep
from ml_template.features.group_statistics import GroupStatisticsStep
from ml_template.features.target_encoding import TargetEncodingStep
from ml_template.preprocessing.numeric_preprocessing import NumericPreprocessingStep

STEP_TYPES: dict[str, type] = {
    PowerTransformStep.name: PowerTransformStep,
    GroupStatisticsStep.name: GroupStatisticsStep,
    TargetEncodingStep.name: TargetEncodingStep,
    NumericPreprocessingStep.name: NumericPreprocessingStep,
}


def build_steps(specs: Sequence[Mapping[str, Any]]) -> list[Any]:
    steps = []
    for spec in specs:
        kind = spec.get("step")
        if kind not in STEP_TYPES:
            known = sorted(STEP_TYPES)
            raise ValueError(f"unknown fitted feature step {kind!r}; available: {known}")
        kwargs = {k: v for k, v in spec.items() if k != "step"}
        try:
            steps.append(STEP_TYPES[kind](**kwargs))
        except TypeError as exc:
            raise ValueError(f"bad {kind} spec {dict(spec)}: {exc}") from exc
    replacements = [i for i, step in enumerate(steps) if getattr(step, "replacement_stage", False)]
    if replacements and replacements != [len(steps) - 1]:
        raise ValueError("numeric_preprocessing must occur once, as the final fitted step")
    return steps


def validate_specs(specs: Sequence[Mapping[str, Any]]) -> None:
    build_steps(specs)


class FittedFeatureTransformer:
    """Ordered steps, fitted together on one frame; ``transform`` appends the new columns."""

    def __init__(self, specs: Sequence[Mapping[str, Any]]) -> None:
        self.specs = [dict(s) for s in specs]
        all_steps = build_steps(self.specs)
        self.preprocessing = (
            all_steps.pop()
            if all_steps and getattr(all_steps[-1], "replacement_stage", False)
            else None
        )
        self.steps = all_steps

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> FittedFeatureTransformer:
        """Fit frozen inference mappings; use fit_transform for model-training inputs."""
        for step in self.steps:
            if getattr(step, "requires_y", False):
                step.fit(X, y)
            else:
                step.fit(X)
        preprocessing = getattr(self, "preprocessing", None)
        if preprocessing is not None:
            preprocessing.fit(self._append(X, [step.transform(X) for step in self.steps]))
        return self

    def _append(self, X: pd.DataFrame, outputs: Sequence[pd.DataFrame]) -> pd.DataFrame:
        for step, frame in zip(self.steps, outputs, strict=True):
            if not isinstance(frame, pd.DataFrame) or not frame.index.equals(X.index):
                raise ValueError("fitted step must return a DataFrame with the same index")
            if list(frame.columns) != step.output_columns():
                raise ValueError("fitted step output columns differ from its declared schema")
        out = pd.concat([X, *outputs], axis=1)
        dup = sorted(set(out.columns[out.columns.duplicated()]))
        if dup:
            raise ValueError(f"fitted features duplicate existing columns {dup}")
        return out

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> pd.DataFrame:
        """Training representation: supervised outputs must be inner out-of-fold."""
        outputs = []
        for step in self.steps:
            if getattr(step, "requires_y", False):
                outputs.append(step.fit_transform(X, y))
            else:
                step.fit(X)
                outputs.append(step.transform(X))
        out = self._append(X, outputs)
        preprocessing = getattr(self, "preprocessing", None)
        if preprocessing is not None:
            preprocessing.fit(out)
            out = preprocessing.transform(out)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._append(X, [step.transform(X) for step in self.steps])
        preprocessing = getattr(self, "preprocessing", None)
        return out if preprocessing is None else preprocessing.transform(out)

    def output_columns(self, input_columns: Sequence[str]) -> list[str]:
        columns = [*input_columns, *(c for s in self.steps for c in s.output_columns())]
        preprocessing = getattr(self, "preprocessing", None)
        return columns if preprocessing is None else columns + preprocessing.output_columns()

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> FittedFeatureTransformer:
        return joblib.load(path)


def transformer_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + ".transformer.joblib")


class TransformingAdapter:
    """ModelAdapter wrapper: fit the feature steps on the rows the inner model trains on."""

    def __init__(self, inner: Any, specs: Sequence[Mapping[str, Any]]) -> None:
        self.inner = inner
        self.specs = [dict(s) for s in specs]
        self.transformer: FittedFeatureTransformer | None = None

    @property
    def max_rounds(self) -> int:
        return self.inner.max_rounds

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int:
        self.transformer = FittedFeatureTransformer(self.specs)
        train = self.transformer.fit_transform(X_train, y_train)
        return self.inner.fit_with_validation(
            train, y_train, self.transformer.transform(X_val), y_val
        )

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None:
        self.transformer = FittedFeatureTransformer(self.specs)
        train = self.transformer.fit_transform(X, y)
        self.inner.refit(train, y, n_rounds)

    def learning_curve(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_rounds: int,
        metric: str,
    ) -> pd.DataFrame:
        transformer = FittedFeatureTransformer(self.specs)
        train = transformer.fit_transform(X_train, y_train)
        return self.inner.learning_curve(
            train,
            y_train,
            transformer.transform(X_val),
            y_val,
            n_rounds,
            metric,
        )

    @property
    def early_stopping_history(self):
        return getattr(self.inner, "early_stopping_history", None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        if not callable(getattr(self.inner, "predict", None)):
            raise TypeError("inner adapter does not support regression prediction")
        return self.inner.predict(self.transformer.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        return self.inner.predict_proba(self.transformer.transform(X))

    def estimator_feature_names(self) -> list[str]:
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        return self.inner.estimator_feature_names()

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        if not callable(getattr(self.inner, "predict_margin", None)):
            raise TypeError("inner adapter does not support margin prediction")
        return self.inner.predict_margin(self.transformer.transform(X))

    def explain_margin(self, X: pd.DataFrame, *, tolerance=None):
        """Transform exactly once; returned features are the estimator's expanded inputs."""
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        if not callable(getattr(self.inner, "explain_margin", None)):
            raise TypeError("inner adapter does not support margin explanations")
        transformed = self.transformer.transform(X)
        expected = self.transformer.output_columns(list(X.columns))
        if list(transformed.columns) != expected:
            raise ValueError("fitted transformer output order mismatch")
        return self.inner.explain_margin(transformed, tolerance=tolerance)

    def save(self, path: Path) -> None:
        if self.transformer is None:
            raise RuntimeError("model is not fitted")
        self.inner.save(path)
        self.transformer.save(transformer_path(path))

    @classmethod
    def load(
        cls, inner: Any, specs: Sequence[Mapping[str, Any]], path: Path
    ) -> TransformingAdapter:
        adapter = cls(inner, specs)
        adapter.transformer = FittedFeatureTransformer.load(transformer_path(path))
        from ml_template.preprocessing.numeric_preprocessing import validate_saved_specs

        validate_saved_specs(adapter.transformer, specs)
        return adapter
