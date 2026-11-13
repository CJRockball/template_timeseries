import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from ml_template.models.model_factory import prepare_features  # noqa: E402
from ml_template.models.xgb_classifier import XGBAdapter, resolve_scale_pos_weight  # noqa: E402
from ml_template.training.run_config import XGBParams  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "type": rng.choice(["L", "M", "H"], size=n),
        }
    )
    y = pd.Series(((X["a"] + (X["type"] == "H") + rng.normal(0, 0.5, n)) > 1.4).astype(int))
    return prepare_features(X), y


def test_scale_pos_weight_resolution():
    y = pd.Series([0] * 90 + [1] * 10)
    assert resolve_scale_pos_weight("auto", y) == pytest.approx(9.0)
    assert resolve_scale_pos_weight(3, y) == 3.0
    with pytest.raises(ValueError):
        resolve_scale_pos_weight("auto", pd.Series([0, 0]))


def test_prepare_features_makes_text_categorical_once():
    X = pd.DataFrame({"t": ["L", "M"], "v": [1.0, 2.0]})
    out = prepare_features(X)
    assert isinstance(out["t"].dtype, pd.CategoricalDtype) and out["v"].dtype == float
    # input untouched (its text dtype is object or str, depending on the pandas version)
    assert not isinstance(X["t"].dtype, pd.CategoricalDtype)
    assert prepare_features(out)["t"].dtype == out["t"].dtype


def test_early_stopping_returns_rounds_within_cap(data):
    X, y = data
    params = XGBParams(n_estimators=300, early_stopping_rounds=10, learning_rate=0.2)
    adapter = XGBAdapter(params, seed=1)
    best = adapter.fit_with_validation(X.iloc[:400], y.iloc[:400], X.iloc[400:], y.iloc[400:])
    assert 1 <= best <= 300
    assert adapter.predict_proba(X.iloc[400:]).shape == (200,)


def test_refit_uses_exact_round_count(data):
    X, y = data
    adapter = XGBAdapter(XGBParams(n_estimators=300), seed=1)
    adapter.refit(X, y, n_rounds=7)
    assert adapter.model.get_booster().num_boosted_rounds() == 7


def test_same_seed_same_predictions(data):
    X, y = data
    p = XGBParams(n_estimators=50)
    a, b = XGBAdapter(p, 3), XGBAdapter(p, 3)
    a.refit(X, y, 20)
    b.refit(X, y, 20)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_save_load_round_trip_and_importances(data, tmp_path):
    X, y = data
    p = XGBParams(n_estimators=50)
    adapter = XGBAdapter(p, 3)
    adapter.refit(X, y, 20)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    loaded = XGBAdapter.load(p, 3, path)
    np.testing.assert_allclose(adapter.predict_proba(X), loaded.predict_proba(X), rtol=1e-5)
    assert set(adapter.feature_importances()) == {"a", "b", "type"}
    assert set(loaded.feature_importances()) == {"a", "b", "type"}


def test_unfitted_adapter_raises(data, tmp_path):
    X, _ = data
    adapter = XGBAdapter(XGBParams(), 1)
    with pytest.raises(RuntimeError):
        adapter.predict_proba(X)
    with pytest.raises(RuntimeError):
        adapter.save(tmp_path / "m.ubj")
