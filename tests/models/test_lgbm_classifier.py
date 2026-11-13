import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from ml_template.models import lgbm_classifier as module
from ml_template.models.lgbm_classifier import LGBMAdapter  # noqa: E402
from ml_template.training.run_config import LGBMParams  # noqa: E402

PARAMS = LGBMParams(n_estimators=60, early_stopping_rounds=10)


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


def test_fit_refit_predict_and_save_load(tmp_path):
    X, y = _data()
    train, val = slice(0, 300), slice(300, 400)
    best = LGBMAdapter(PARAMS, seed=1).fit_with_validation(
        X.iloc[train], y.iloc[train], X.iloc[val], y.iloc[val]
    )
    assert 1 <= best <= PARAMS.n_estimators
    model = LGBMAdapter(PARAMS, seed=1)
    model.refit(X.iloc[train], y.iloc[train], best)
    p = model.predict_proba(X.iloc[val])
    assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all()
    path = tmp_path / "model.ubj"
    model.save(path)
    loaded = LGBMAdapter.load(PARAMS, 1, path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[val]), p, rtol=1e-6)
    assert set(loaded.feature_importances()) == {"a", "b", "c"}


def test_learning_curve_table_shape():
    X, y = _data()
    curve = LGBMAdapter(PARAMS, seed=1).learning_curve(
        X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:], 15, "logloss"
    )
    assert set(curve["dataset"]) == {"train", "val"} and set(curve["metric"]) == {"logloss"}
    assert curve.groupby("dataset")["round"].max().tolist() == [15, 15]
    train = curve[curve["dataset"] == "train"]["value"].to_numpy()
    assert train[-1] < train[0]


def test_unfitted_model_fails():
    with pytest.raises(RuntimeError, match="not fitted"):
        LGBMAdapter(PARAMS, seed=1).predict_proba(_data()[0])


def test_eval_args_follow_the_installed_version(monkeypatch):
    pairs = [("X1", "y1"), ("X2", "y2")]
    monkeypatch.setattr(module, "_NEW_EVAL_ARGS", True)
    assert module.eval_args(pairs) == {"eval_X": ("X1", "X2"), "eval_y": ("y1", "y2")}
    monkeypatch.setattr(module, "_NEW_EVAL_ARGS", False)
    assert module.eval_args(pairs) == {"eval_set": pairs}


def test_fit_does_not_use_deprecated_eval_set():
    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LGBMAdapter(PARAMS, seed=1).fit_with_validation(
            X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:]
        )
        LGBMAdapter(PARAMS, seed=1).learning_curve(
            X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:], 10, "logloss"
        )
    assert not [w for w in caught if "eval_set" in str(w.message)]
