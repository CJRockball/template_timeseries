import numpy as np
import pandas as pd
import pytest

from ml_template.features.fitted_features import TransformingAdapter
from ml_template.models.xgb_classifier import XGBAdapter
from ml_template.training.run_config import XGBParams


def data():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {"x": rng.normal(size=120), "kind": pd.Categorical(np.tile(["L", "M", "H"], 40))},
        index=pd.Index(np.arange(1000, 1120), name="entity"),
    )
    y = pd.Series(((X.x > 0.3) | (X.kind == "H")).astype(int), index=X.index)
    return X, y


def test_native_categorical_refit_and_saved_booster(tmp_path):
    X, y = data()
    params = XGBParams(n_estimators=20, max_depth=2, scale_pos_weight=1.0, device="cpu")
    adapter = XGBAdapter(params, 17)
    adapter.refit(X, y, 12)
    before = adapter.explain_margin(X)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    loaded = XGBAdapter.load(params, 17, path)
    after = loaded.explain_margin(X)
    np.testing.assert_allclose(before.contributions, after.contributions)
    np.testing.assert_allclose(after.rows.p_raw, loaded.predict_proba(X))
    np.testing.assert_allclose(after.rows.margin, loaded.predict_margin(X))
    assert after.verification["passed"]
    assert loaded.estimator_feature_names() == list(X.columns)
    pieces = [loaded.explain_margin(X.iloc[i : i + 7]).contributions for i in range(0, len(X), 7)]
    pd.testing.assert_frame_equal(after.contributions, pd.concat(pieces))
    reverse = loaded.explain_margin(X.iloc[::-1])
    pd.testing.assert_frame_equal(
        after.contributions.sort_index(), reverse.contributions.sort_index()
    )
    assert len(loaded.explain_margin(X.iloc[:1]).rows) == 1


def test_live_early_stopping_tree_range():
    X, y = data()
    params = XGBParams(
        n_estimators=60, early_stopping_rounds=3, max_depth=2, scale_pos_weight=1.0, device="cpu"
    )
    adapter = XGBAdapter(params, 17)
    adapter.fit_with_validation(X.iloc[:80], y.iloc[:80], X.iloc[80:], y.iloc[80:])
    out = adapter.explain_margin(X.iloc[80:])
    np.testing.assert_allclose(out.rows.p_raw, adapter.predict_proba(X.iloc[80:]), atol=1e-6)


def test_saved_fitted_statistics_end_to_end(tmp_path):
    X, y = data()
    params = XGBParams(n_estimators=12, max_depth=2, scale_pos_weight=1.0, device="cpu")
    specs = [
        {
            "step": "group_statistics",
            "group_by": ["kind"],
            "columns": ["x"],
            "stats": ["mean", "std"],
            "deviation": True,
            "min_group_size": 2,
        }
    ]
    adapter = TransformingAdapter(XGBAdapter(params, 17), specs)
    adapter.refit(X, y, 12)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    loaded = TransformingAdapter.load(XGBAdapter.load(params, 17, path), specs, path)
    out = loaded.explain_margin(X)
    assert len(out.features.columns) > len(X.columns)
    assert list(out.features.columns) == loaded.estimator_feature_names()
    np.testing.assert_allclose(out.rows.p_raw, loaded.predict_proba(X), atol=1e-6)


def test_wrong_feature_order_and_unfitted_rejected():
    X, y = data()
    adapter = XGBAdapter(XGBParams(scale_pos_weight=1.0, device="cpu"), 17)
    with pytest.raises(RuntimeError):
        adapter.explain_margin(X)
    adapter.refit(X, y, 3)
    with pytest.raises(ValueError, match="order"):
        adapter.explain_margin(X[["kind", "x"]])
