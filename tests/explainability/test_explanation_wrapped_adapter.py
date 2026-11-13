from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ml_template.explainability.result import build_margin_explanation
from ml_template.features import fitted_features as module

TransformingAdapter = module.TransformingAdapter


class AppendTransformer:
    def __init__(self):
        self.calls = 0

    def transform(self, X):
        self.calls += 1
        assert "derived" not in X
        out = X.copy()
        out["derived"] = out["x"] * 2
        return out

    def output_columns(self, columns):
        return [*columns, "derived"]

    def fit(self, *args):
        raise AssertionError("explanation must not fit transformations")


class ToyInner:
    def estimator_feature_names(self):
        return ["x", "derived"]

    def predict_margin(self, X):
        return X.sum(axis=1).to_numpy() - 1

    def explain_margin(self, X, *, tolerance=None):
        margin = self.predict_margin(X)
        return build_margin_explanation(
            X, X.to_numpy(), np.full(len(X), -1.0), margin, expit(margin), tolerance=tolerance
        )


def test_wrapper_transforms_once_and_returns_expanded_features():
    adapter = TransformingAdapter(ToyInner(), [])
    adapter.transformer = AppendTransformer()
    X = pd.DataFrame({"x": [1.0, 2.0]}, index=[10, 20])
    out = adapter.explain_margin(X)
    assert adapter.transformer.calls == 1
    assert out.features.columns.tolist() == ["x", "derived"]
    assert adapter.estimator_feature_names() == ["x", "derived"]
    assert list(X.columns) == ["x"]


def test_wrapper_margin_transforms_once():
    adapter = TransformingAdapter(ToyInner(), [])
    adapter.transformer = AppendTransformer()
    np.testing.assert_allclose(adapter.predict_margin(pd.DataFrame({"x": [1.0]})), [2])
    assert adapter.transformer.calls == 1


@pytest.mark.parametrize("method", ["predict_margin", "explain_margin", "estimator_feature_names"])
def test_wrapper_unfitted_rejected(method):
    adapter = TransformingAdapter(ToyInner(), [])
    with pytest.raises(RuntimeError, match="not fitted"):
        if method == "estimator_feature_names":
            getattr(adapter, method)()
        else:
            getattr(adapter, method)(pd.DataFrame({"x": [1.0]}))


def test_wrapper_unsupported_adapter_rejected():
    adapter = TransformingAdapter(SimpleNamespace(), [])
    adapter.transformer = AppendTransformer()
    with pytest.raises(TypeError):
        adapter.explain_margin(pd.DataFrame({"x": [1.0]}))


def test_real_saved_group_statistics(tmp_path):
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
    X = pd.DataFrame(
        {"kind": pd.Categorical(["L", "L", "H", "H"]), "x": [1.0, 2.0, 4.0, 8.0]},
        index=[1, 2, 3, 4],
    )
    transformer = module.FittedFeatureTransformer(specs).fit(X)
    path = tmp_path / "transformer.joblib"
    transformer.save(path)
    loaded = module.FittedFeatureTransformer.load(path)
    pd.testing.assert_frame_equal(transformer.transform(X), loaded.transform(X))
    assert loaded.output_columns(list(X.columns)) == list(loaded.transform(X).columns)
