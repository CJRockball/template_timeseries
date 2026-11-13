import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_template.evaluation import permutation_importance as pi


class LinearAdapter:
    """Uses only 'signal' and 'noise'; 'kind' (categorical) is ignored by the model."""

    def __init__(self, X, y):
        self.cols = ["signal", "noise"]
        self.model = LogisticRegression(max_iter=300).fit(X[self.cols], y)
        self.seen_dtypes: list = []
        self.seen_rows: list = []

    def predict_proba(self, X):
        self.seen_dtypes.append(X["kind"].dtype)
        self.seen_rows.append(len(X))
        return self.model.predict_proba(X[self.cols])[:, 1]


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    y = (signal + 0.7 * rng.normal(size=n) > 1.0).astype(int)
    idx = pd.Index(np.arange(n), name="key")
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "kind": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        },
        index=idx,
    )
    return X, pd.Series(y, index=idx)


def test_signal_matters_noise_does_not_and_ignored_column_is_zero():
    X, y = _data()
    adapter = LinearAdapter(X, y)
    table = pi.permutation_importance(adapter, X, y, n_repeats=10, seed=1).set_index("feature")
    assert table.loc["signal", "mean"] > 0.2
    assert abs(table.loc["noise", "mean"]) < 0.03
    assert table.loc["kind", "mean"] == 0.0  # the model never reads it
    assert table.loc["signal", "mean"] > 5 * abs(table.loc["noise", "mean"])


def test_categorical_dtype_is_preserved_when_shuffling():
    X, y = _data(n=400)
    adapter = LinearAdapter(X, y)
    pi.permutation_importance(adapter, X, y, n_repeats=2, seed=1)
    assert all(isinstance(d, pd.CategoricalDtype) for d in adapter.seen_dtypes)


def test_deterministic_and_input_not_modified():
    X, y = _data(n=600)
    before = X.copy()
    adapter = LinearAdapter(X, y)
    a = pi.permutation_importance(adapter, X, y, n_repeats=5, seed=3)
    b = pi.permutation_importance(adapter, X, y, n_repeats=5, seed=3)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(X, before)


def test_single_repeat_has_nan_std_and_cap_limits_rows():
    X, y = _data(n=2000)
    adapter = LinearAdapter(X, y)
    table = pi.permutation_importance(adapter, X, y, n_repeats=1, seed=1, max_rows=500)
    assert table["std"].isna().all()
    assert max(adapter.seen_rows) <= 500


def test_single_class_and_duplicate_index_rejected():
    X, y = _data(n=300)
    adapter = LinearAdapter(X, y)
    with pytest.raises(ValueError, match="single class"):
        pi.permutation_importance(adapter, X, y * 0, n_repeats=2)
    dup = pd.concat([X, X.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        pi.permutation_importance(adapter, dup, pd.concat([y, y.iloc[:1]]))


def test_rows_for_run_have_per_fold_and_pooled_rows():
    X, y = _data(n=1800)
    folds = {k: np.arange(k * 600, (k + 1) * 600) for k in range(3)}
    keys = {k: X.index[idx].to_numpy() for k, idx in folds.items()}
    models = {k: LinearAdapter(X, y) for k in folds}
    rows = pi.importance_rows_for_run(models, keys, X, y, n_repeats=3, seed=5, max_rows=None)
    df = pd.DataFrame(rows)
    assert set(df["method"]) == {pi.METHOD} and set(df["split"]) == {"cv"}
    assert set(df["fold"]) == {-1, 0, 1, 2}
    assert len(df) == 4 * 3  # 3 features x (3 folds + pooled)
    for feature in X.columns:
        per_fold = df[(df["feature"] == feature) & (df["fold"] >= 0)]["value"].to_numpy()
        pooled = df[(df["feature"] == feature) & (df["fold"] == -1)].iloc[0]
        assert pooled["value"] == pytest.approx(per_fold.mean())
        assert pooled["std"] == pytest.approx(np.std(per_fold, ddof=1))
    assert not any(math.isnan(v) for v in df["value"])
