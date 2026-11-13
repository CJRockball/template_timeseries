"""Binary target encoding with label-independent inner cross-fitting.

fit_transform(X, y) supplies out-of-fold TRAINING encodings and retains mappings
fitted on all supplied rows for subsequent transform(X_new). fit(X, y).transform(X)
is NOT a substitute for the cross-fitted training representation. No labels,
training frame, or out-of-fold training matrix are kept in the fitted object.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class TargetEncodingStep:
    """Append one smoothed target mean for a single or composite categorical key."""

    name = "target_encoding"
    requires_y = True

    def __init__(
        self,
        group_by: Sequence[str],
        smoothing: float = 20.0,
        n_splits: int = 5,
        seed: int = 42,
        min_group_size: int = 1,
        output_name: str | None = None,
    ) -> None:
        if isinstance(group_by, (str, bytes)) or not isinstance(group_by, Sequence):
            raise ValueError("group_by must be a non-empty sequence of column names")
        if not group_by or any(not isinstance(c, str) or not c for c in group_by):
            raise ValueError("group_by must contain non-empty column names")
        if len(set(group_by)) != len(group_by):
            raise ValueError("group_by contains duplicate names")
        if isinstance(smoothing, bool) or not isinstance(smoothing, Real):
            raise ValueError("smoothing must be a finite number >= 0")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be a finite number >= 0")
        for label, value, minimum in (
            ("n_splits", n_splits, 2),
            ("min_group_size", min_group_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if output_name is not None and (not isinstance(output_name, str) or not output_name):
            raise ValueError("output_name must be a non-empty string or None")
        self.group_by = list(group_by)
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.min_group_size = int(min_group_size)
        self.output_name = output_name or "te__" + "__".join(self.group_by)

    def output_columns(self) -> list[str]:
        return [self.output_name]

    def _keys(self, X: pd.DataFrame) -> pd.MultiIndex:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not X.index.is_unique or X.columns.duplicated().any():
            raise ValueError("X must have unique row keys and column names")
        missing = [c for c in self.group_by if c not in X.columns]
        if missing:
            raise KeyError(f"target_encoding needs columns {missing}")
        arrays = []
        for column in self.group_by:
            s = X[column]
            # A separate null flag prevents collisions with every literal sentinel string.
            arrays.extend((s.isna().to_numpy(), s.astype(object).where(s.notna(), "").to_numpy()))
        return pd.MultiIndex.from_arrays(arrays)

    @staticmethod
    def _target(X: pd.DataFrame, y: pd.Series | np.ndarray | None) -> np.ndarray:
        if y is None:
            raise ValueError("target_encoding requires training labels y")
        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("X and y must share the same index in the same order")
        a = np.asarray(y)
        if a.ndim != 1 or len(a) != len(X) or len(a) == 0:
            raise ValueError("y must be non-empty, one-dimensional and aligned with X")
        if pd.isna(a).any() or not pd.Series(a).isin([0, 1]).all():
            raise ValueError("target_encoding requires non-missing binary labels 0/1")
        return a.astype(float)

    def _learn(self, keys: pd.MultiIndex, y: np.ndarray) -> tuple[pd.Series, float]:
        prior = float(y.mean())
        values = pd.Series(y, index=keys)
        table = values.groupby(level=list(range(keys.nlevels)), sort=False).agg(["sum", "count"])
        mean = (table["sum"] + self.smoothing * prior) / (table["count"] + self.smoothing)
        mean = mean.where(table["count"] >= self.min_group_size, prior)
        return mean, prior

    @staticmethod
    def _apply(keys: pd.MultiIndex, mapping: pd.Series, prior: float) -> np.ndarray:
        return mapping.reindex(keys).fillna(prior).to_numpy(dtype=float)

    def _frame(self, X: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("target encoding produced invalid probabilities")
        return pd.DataFrame({self.output_name: values}, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> TargetEncodingStep:
        keys = self._keys(X)
        target = self._target(X, y)
        self.mapping_, self.prior_ = self._learn(keys, target)
        self.n_train_ = len(X)
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> pd.DataFrame:
        keys = self._keys(X)
        target = self._target(X, y)
        if len(X) < self.n_splits:
            raise ValueError("target_encoding needs at least n_splits training rows")
        encoded = np.empty(len(X), dtype=float)
        # KFold does not inspect y: changing a row's label cannot alter its assignment.
        splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for fit, val in splitter.split(np.arange(len(X))):
            mapping, prior = self._learn(keys[fit], target[fit])
            encoded[val] = self._apply(keys[val], mapping, prior)
        # Only the full fitting partition's mappings persist for inference.
        self.mapping_, self.prior_ = self._learn(keys, target)
        self.n_train_ = len(X)
        return self._frame(X, encoded)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("target encoder is not fitted")
        return self._frame(X, self._apply(self._keys(X), self.mapping_, self.prior_))
