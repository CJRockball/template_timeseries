"""Target checks, explicit/inferred roles and CPU batch index generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_labels(X, y):
    if not isinstance(X, pd.DataFrame) or len(X) == 0 or not X.columns.is_unique:
        raise ValueError("training features must be a nonempty DataFrame")
    if not isinstance(y, pd.Series) or not X.index.equals(y.index):
        raise ValueError("labels must be a Series aligned with feature rows")
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("each training/validation slice requires both nonmissing binary classes")
    return y.to_numpy(dtype=np.float32, copy=True)


def resolve_preprocessing(params, X):
    from ml_template.preprocessing.nn_config import NNPreprocessorConfig

    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("features must have unique columns")
    if any(not isinstance(c, str) for c in X):
        raise ValueError("feature names must be strings")
    config = params.preprocessing
    numeric, categorical = config.numeric, config.categorical
    if numeric is None and categorical is None:
        categorical = tuple(
            c
            for c in X
            if (
                isinstance(X[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[c].dtype)
                or pd.api.types.is_object_dtype(X[c].dtype)
            )
        )
        numeric = tuple(c for c in X if c not in categorical)
    elif numeric is None:
        numeric = tuple(c for c in X if c not in categorical)
    elif categorical is None:
        categorical = tuple(c for c in X if c not in numeric)
    if set(numeric) | set(categorical) != set(X.columns):
        raise ValueError("configured NN features do not cover the received columns exactly")
    raw = config.model_dump(exclude={"numeric", "categorical"})
    return NNPreprocessorConfig(numeric=numeric, categorical=categorical, **raw)


def batch_indices(n_rows, batch_size, rng=None):
    if n_rows < 0 or batch_size < 1:
        raise ValueError("invalid batch dimensions")
    order = np.arange(n_rows) if rng is None else rng.permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        yield order[start : start + batch_size]


def resolve_pos_weight(setting, labels):
    if setting == "auto":
        positives = int(np.sum(labels))
        negatives = len(labels) - positives
        if not positives or not negatives:
            raise ValueError("automatic class weight requires both classes")
        return negatives / positives
    return float(setting)
