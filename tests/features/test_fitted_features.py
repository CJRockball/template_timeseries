import numpy as np
import pandas as pd
import pytest

from ml_template.features.fitted_features import (
    FittedFeatureTransformer,
    TransformingAdapter,
    build_steps,
    transformer_path,
    validate_specs,
)

GROUP = {"step": "group_statistics", "group_by": ["type"], "columns": ["v"], "stats": ["mean"]}


def _frame(types, values, start=1):
    idx = pd.Index(range(start, start + len(types)), name="udi")
    return pd.DataFrame({"type": pd.Categorical(types), "v": values}, index=idx)


class FakeInner:
    max_rounds = 100

    def __init__(self):
        self.fit_seen = None
        self.refit_seen = None
        self.predict_seen = None

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        self.fit_seen = (X_train.copy(), X_val.copy())
        return 7

    def refit(self, X, y, n_rounds):
        self.refit_seen = X.copy()

    def predict_proba(self, X):
        self.predict_seen = X.copy()
        return np.full(len(X), 0.5)

    def save(self, path):
        path.write_text("model")


def test_validation_rows_use_training_statistics():
    train = _frame(["L", "L", "M"], [1.0, 3.0, 10.0])
    val = _frame(["L"], [100.0], start=10)
    y = pd.Series([0, 1, 0], index=train.index)
    adapter = TransformingAdapter(FakeInner(), [GROUP])
    assert adapter.fit_with_validation(train, y, val, pd.Series([0], index=val.index)) == 7
    fit_train, fit_val = adapter.inner.fit_seen
    assert fit_train["v__by_type__mean"].tolist() == [2.0, 2.0, 10.0]
    assert fit_val["v__by_type__mean"].tolist() == [2.0]  # 100.0 did not enter the mean


def test_refit_fits_on_its_own_rows_and_predict_applies_them():
    train = _frame(["L", "L", "M"], [1.0, 3.0, 10.0])
    adapter = TransformingAdapter(FakeInner(), [GROUP])
    adapter.refit(train, pd.Series([0, 1, 0], index=train.index), 5)
    assert adapter.inner.refit_seen["v__by_type__mean"].tolist() == [2.0, 2.0, 10.0]
    new = _frame(["M", "H"], [999.0, 999.0], start=20)
    adapter.predict_proba(new)
    expected = [10.0, train["v"].mean()]
    assert adapter.inner.predict_seen["v__by_type__mean"].tolist() == expected
    assert adapter.max_rounds == 100


def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="not fitted"):
        TransformingAdapter(FakeInner(), [GROUP]).predict_proba(_frame(["L"], [1.0]))


def test_save_and_load_round_trip(tmp_path):
    train = _frame(["L", "L", "M"], [1.0, 3.0, 10.0])
    adapter = TransformingAdapter(FakeInner(), [GROUP])
    adapter.refit(train, pd.Series([0, 1, 0], index=train.index), 5)
    path = tmp_path / "fold_0.ubj"
    adapter.save(path)
    assert transformer_path(path).name == "fold_0.transformer.joblib"
    assert transformer_path(path).exists()
    loaded = TransformingAdapter.load(FakeInner(), [GROUP], path)
    new = _frame(["M"], [0.0], start=30)
    loaded.predict_proba(new)
    assert loaded.inner.predict_seen["v__by_type__mean"].tolist() == [10.0]


def test_power_transform_rules_and_rank_preservation():
    df = _frame(["L"] * 5, [1.0, 2.0, 4.0, 8.0, 100.0])
    spec = {"step": "power_transform", "columns": ["v"], "method": "yeo-johnson"}
    out = FittedFeatureTransformer([spec]).fit(df).transform(df)
    assert np.isfinite(out["pt_v"]).all()
    assert out["pt_v"].rank().tolist() == df["v"].rank().tolist()
    bad = df.assign(v=[1.0, 0.0, 2.0, 3.0, 4.0])
    box = {"step": "power_transform", "columns": ["v"], "method": "box-cox"}
    with pytest.raises(ValueError, match="strictly positive"):
        FittedFeatureTransformer([box]).fit(bad)
    fitted = FittedFeatureTransformer([box]).fit(df)
    with pytest.raises(ValueError, match="strictly positive"):
        fitted.transform(bad)


def test_specs_are_validated():
    with pytest.raises(ValueError, match="unknown fitted feature step"):
        validate_specs([{"step": "nope"}])
    with pytest.raises(ValueError, match="bad group_statistics"):
        build_steps([{"step": "group_statistics", "group_by": ["type"], "columns": ["v"], "x": 1}])
    clash = {"step": "group_statistics", "group_by": ["type"], "columns": ["v"], "stats": ["mean"]}
    df = _frame(["L", "M"], [1.0, 2.0]).assign(v__by_type__mean=[0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        FittedFeatureTransformer([clash]).fit(df).transform(df)
