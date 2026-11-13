import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from ml_template.models.catboost_classifier import CatBoostAdapter, prepare  # noqa: E402
from ml_template.training.run_config import CatBoostParams  # noqa: E402

PARAMS = CatBoostParams(n_estimators=60, early_stopping_rounds=10)


def _data(n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": pd.Categorical(rng.choice(["L", "M", "H"], size=n)),
        }
    )
    y = pd.Series((X["a"] + (X["c"] == "H") + rng.normal(scale=0.5, size=n) > 1).astype(int))
    return X, y


def test_prepare_converts_only_non_numeric_columns():
    X, _ = _data(5)
    out, cats = prepare(X)
    assert cats == ["c"] and out["c"].map(type).eq(str).all()
    assert out["a"].dtype == X["a"].dtype


def test_fit_refit_predict_and_save_load(tmp_path):
    X, y = _data()
    train, val = slice(0, 300), slice(300, 400)
    best = CatBoostAdapter(PARAMS, seed=1).fit_with_validation(
        X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val]
    )
    assert 1 <= best <= PARAMS.n_estimators
    model = CatBoostAdapter(PARAMS, seed=1)
    model.refit(X.iloc[train], y.iloc[train], best)
    p = model.predict_proba(X.iloc[val])
    assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all()
    path = tmp_path / "model.ubj"
    model.save(path)
    loaded = CatBoostAdapter.load(PARAMS, 1, path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[val]), p, rtol=1e-6)
    assert set(loaded.feature_importances()) == {"a", "b", "c"}


def test_learning_curve_table_shape():
    X, y = _data()
    curve = CatBoostAdapter(PARAMS, seed=1).learning_curve(
        X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:], 15, "logloss"
    )
    assert set(curve["dataset"]) == {"train", "val"} and set(curve["metric"]) == {"logloss"}
    assert curve.groupby("dataset")["round"].max().tolist() == [15, 15]
    train = curve[curve["dataset"] == "train"]["value"].to_numpy()
    assert train[-1] < train[0]


def test_unfitted_model_fails():
    with pytest.raises(RuntimeError, match="not fitted"):
        CatBoostAdapter(PARAMS, seed=1).predict_proba(_data()[0])
