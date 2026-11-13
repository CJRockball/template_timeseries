"""Synthetic contracts for reused numeric preprocessing and fitted adapters."""

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_template.features.fitted_features import FittedFeatureTransformer, TransformingAdapter
from ml_template.preprocessing.numeric_preprocessing import NumericPreprocessingStep


@pytest.mark.parametrize("strategy,expected", [("mean", 4.0), ("median", 2.0), ("constant", -7.0)])
def test_reuses_observed_training_statistics(strategy, expected):
    train = pd.DataFrame({"a": [1.0, 2.0, 9.0, np.nan]})
    step = NumericPreprocessingStep(["a"], numeric_strategy=strategy, constant_fill=-7)
    step.fit(train)
    state = step.preprocessor.state_dict()
    out = step.transform(pd.DataFrame({"a": [np.nan, 100000.0]}))
    assert out.a.iloc[0] == pytest.approx(expected)
    assert out.a__is_missing.tolist() == [1.0, 0.0]
    assert step.preprocessor.state_dict() == state
    assert train.a.isna().iloc[-1]


def test_scaling_uses_observed_values_not_filled_variance():
    step = NumericPreprocessingStep(["a"], numeric_strategy="mean", scaling="standard")
    out = step.fit(pd.DataFrame({"a": [1.0, 3.0, np.nan]})).transform(
        pd.DataFrame({"a": [1.0, 3.0, np.nan]})
    )
    np.testing.assert_allclose(out.a, [-1.0, 1.0, 0.0])


def test_all_missing_constant_and_nullable_integer():
    X = pd.DataFrame({"a": pd.Series([pd.NA, pd.NA], dtype="Int64"), "b": [7.0, 7.0]})
    step = NumericPreprocessingStep(["a", "b"], scaling="standard", all_missing_fill=-2)
    out = step.fit(X).transform(X)
    np.testing.assert_allclose(out.a, [-2, -2])
    np.testing.assert_allclose(out.b, [0, 0])


def test_preserves_category_dtype_and_unselected_native_nulls():
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan],
            "native": [np.nan, 2.0],
            "k": pd.Categorical(["x", None], categories=["x", "y"]),
        }
    )
    out = NumericPreprocessingStep(["a"]).fit(X).transform(X)
    pd.testing.assert_series_equal(X.k, out.k)
    pd.testing.assert_series_equal(X.native, out.native)


def test_flags_exist_for_new_missingness_and_can_be_disabled():
    X = pd.DataFrame({"a": [1.0, 3.0]})
    assert NumericPreprocessingStep(["a"]).fit(X).transform(X).a__is_missing.eq(0).all()
    step = NumericPreprocessingStep(["a"], missing_indicators=False).fit(X)
    out = step.transform(pd.DataFrame({"a": [np.nan]}))
    assert list(out) == ["a"] and out.a.iloc[0] == 2


@pytest.mark.parametrize("bad", [np.inf, -np.inf, "bad"])
def test_invalid_numeric_inputs(bad):
    with pytest.raises((ValueError, TypeError)):
        NumericPreprocessingStep(["a"]).fit(pd.DataFrame({"a": [bad]}))


def test_schema_order_missing_column_collision_and_not_fitted():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    step = NumericPreprocessingStep(["a"])
    with pytest.raises(RuntimeError):
        step.transform(X)
    step.fit(X)
    for bad in [X[["b", "a"]], X[["b"]], X.assign(c=3)]:
        with pytest.raises(ValueError):
            step.transform(bad)
    with pytest.raises(ValueError):
        NumericPreprocessingStep(["a"]).fit(X.assign(a__is_missing=0))


def test_pickle_and_batch_replay(tmp_path):
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 5.0]}, index=["a", "b", "c", "d"])
    step = NumericPreprocessingStep(["a"]).fit(X)
    path = tmp_path / "pre.joblib"
    joblib.dump(step, path)
    loaded = joblib.load(path)
    whole = step.transform(X)
    pd.testing.assert_frame_equal(whole, loaded.transform(X))
    replay = pd.concat([loaded.transform(X.iloc[i : i + 1]) for i in range(len(X))])
    pd.testing.assert_frame_equal(whole, replay)
    pd.testing.assert_frame_equal(whole, loaded.transform(X.iloc[::-1]).loc[X.index])


def test_transformer_replaces_inputs_and_output_schema(tmp_path):
    specs = [{"step": "numeric_preprocessing", "columns": ["a"]}]
    transformer = FittedFeatureTransformer(specs)
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = transformer.fit_transform(X)
    assert out.a.tolist() == [1.0, 2.0, 3.0]
    assert list(out) == transformer.output_columns(list(X))
    path = tmp_path / "transformer.joblib"
    transformer.save(path)
    pd.testing.assert_frame_equal(out, FittedFeatureTransformer.load(path).transform(X))


def test_preprocessing_must_be_last_and_unique():
    spec = {"step": "numeric_preprocessing", "columns": ["a"]}
    with pytest.raises(ValueError):
        FittedFeatureTransformer([spec, spec])


class RecordingRegressor:
    max_rounds = 4
    early_stopping_history = None

    def fit_with_validation(self, X, y, V, v):
        self.train, self.validation = X.copy(), V.copy()
        self.names = list(X)
        return 2

    def refit(self, X, y, rounds):
        self.train, self.names = X.copy(), list(X)

    def predict(self, X):
        assert list(X) == self.names
        return X.a.to_numpy(float)

    def estimator_feature_names(self):
        return self.names

    def save(self, path):
        joblib.dump(self, path)


def test_wrapper_fit_is_fold_local_and_restores_saved_state(tmp_path):
    specs = [{"step": "numeric_preprocessing", "columns": ["a"]}]
    model = TransformingAdapter(RecordingRegressor(), specs)
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    val = pd.DataFrame({"a": [1000.0, np.nan]})
    assert (
        model.fit_with_validation(train, pd.Series([0.0, 1.0, 2.0]), val, pd.Series([0.0, 1.0]))
        == 2
    )
    assert model.inner.validation.a.iloc[1] == 2
    model.refit(train, pd.Series([0.0, 1.0, 2.0]), 2)
    path = tmp_path / "model.ubj"
    model.save(path)
    loaded = TransformingAdapter.load(joblib.load(path), specs, path)
    np.testing.assert_allclose(model.predict(val), loaded.predict(val))
    wrong = [{"step": "numeric_preprocessing", "columns": ["a"], "scaling": "standard"}]
    with pytest.raises(ValueError, match="specification"):
        TransformingAdapter.load(joblib.load(path), wrong, path)


def test_no_fitted_steps_preserves_native_missingness():
    X = pd.DataFrame({"a": [1.0, np.nan]})
    pd.testing.assert_frame_equal(X, FittedFeatureTransformer([]).fit_transform(X))


class RecordingClassifier(RecordingRegressor):
    def predict_proba(self, X):
        assert list(X) == self.names
        return np.full(len(X), 0.5)


def test_binary_wrapper_interface_is_preserved():
    model = TransformingAdapter(
        RecordingClassifier(), [{"step": "numeric_preprocessing", "columns": ["a"]}]
    )
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    model.refit(X, pd.Series([0, 1, 0]), 2)
    np.testing.assert_allclose(model.predict_proba(X), [0.5, 0.5, 0.5])
