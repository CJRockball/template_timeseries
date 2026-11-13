"""Leakage boundaries and persistence for fitted binary target encoding."""

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.model_selection import KFold

from ml_template.features.fitted_features import (
    FittedFeatureTransformer,
    TransformingAdapter,
    validate_specs,
)
from ml_template.features.target_encoding import TargetEncodingStep

SPEC = {
    "step": "target_encoding",
    "group_by": ["route"],
    "smoothing": 2.0,
    "n_splits": 3,
    "seed": 17,
}


def data():
    X = pd.DataFrame(
        {"route": ["A"] * 6 + ["B"] * 6, "v": np.arange(12, dtype=float) + 1},
        index=pd.Index([19, 4, 27, 2, 80, 9, 32, 6, 45, 8, 99, 10], name="id"),
    )
    y = pd.Series([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], index=X.index)
    return X, y


def encoder(**kwargs):
    return TargetEncodingStep(**{**{k: v for k, v in SPEC.items() if k != "step"}, **kwargs})


def test_smoothing_equation_and_unknown_fallback():
    X, y = data()
    step = encoder().fit(X, y)
    result = step.transform(pd.DataFrame({"route": ["A", "B", "NEW"]}))
    prior = y.mean()
    assert result.iloc[:, 0].tolist() == pytest.approx(
        [(4 + 2 * prior) / 8, (2 + 2 * prior) / 8, prior]
    )


def test_each_training_encoding_uses_only_complement_labels_and_prior():
    X, y = data()
    original = X.copy(deep=True)
    result = encoder().fit_transform(X, y)
    for fit, val in KFold(3, shuffle=True, random_state=17).split(X):
        expected = encoder().fit(X.iloc[fit], y.iloc[fit]).transform(X.iloc[val])
        assert_frame_equal(result.iloc[val], expected)
    assert_frame_equal(X, original)


def test_changing_entire_inner_validation_labels_does_not_change_its_encodings():
    X, y = data()
    result = encoder().fit_transform(X, y)
    for _, val in KFold(3, shuffle=True, random_state=17).split(X):
        changed = y.copy()
        changed.iloc[val] = 1 - changed.iloc[val]
        other = encoder().fit_transform(X, changed)
        assert_frame_equal(result.iloc[val], other.iloc[val])


def test_unique_categories_use_inner_training_prior_not_their_own_labels():
    X, y = data()
    X["route"] = [f"unique-{i}" for i in range(len(X))]
    result = encoder(smoothing=0).fit_transform(X, y)
    for fit, val in KFold(3, shuffle=True, random_state=17).split(X):
        assert np.allclose(result.iloc[val, 0], y.iloc[fit].mean())
    assert not np.array_equal(result.iloc[:, 0], y)


def test_full_mapping_is_retained_after_cross_fitting():
    X, y = data()
    step = encoder()
    training = step.fit_transform(X, y)
    full = encoder().fit(X, y).transform(X)
    assert_frame_equal(step.transform(X), full)
    assert not np.allclose(training, full)
    assert step.n_train_ == len(X)
    assert not any(k in vars(step) for k in ("y_", "X_", "oof_", "training_"))


def test_min_group_size_applies_to_mapping():
    X, y = data()
    step = encoder(min_group_size=7).fit(X, y)
    assert np.allclose(step.transform(X), y.mean())


def test_missing_is_a_real_category_not_a_literal_sentinel():
    X = pd.DataFrame(
        {"a": pd.array([None, None, "", "", "nan", "nan", "<NA>", "<NA>"], dtype="string")}
    )
    y = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    step = encoder(group_by=["a"], smoothing=0).fit(X, y)
    assert step.transform(X).iloc[:, 0].tolist() == y.tolist()
    assert len(step.mapping_) == 4


def test_composite_keys_do_not_have_delimiter_collisions():
    X = pd.DataFrame({"a": ["a|b", "a", "a|b", "a"], "b": ["c", "b|c", "c", "b|c"]})
    y = pd.Series([1, 0, 1, 0])
    step = encoder(group_by=["a", "b"], smoothing=0).fit(X, y)
    assert step.transform(X).iloc[:, 0].tolist() == y.tolist()


@pytest.mark.parametrize("dtype", ["category", "string", "object", "Int64", "int64", "float64"])
def test_key_dtypes_and_nullable_keys(dtype):
    values = [1, 2, 1, 2, 1, 2]
    if dtype in {"category", "string", "object", "Int64"}:
        values[1] = None
    X = pd.DataFrame({"a": pd.array(values, dtype=dtype)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    step = encoder(group_by=["a"])
    assert np.isfinite(step.fit_transform(X, y)).all().all()
    assert_frame_equal(step.transform(X.iloc[:1]), step.transform(X).iloc[:1])


def test_numeric_key_int_and_float_representations_match():
    X = pd.DataFrame({"route": pd.array([100, 100, 200, 200], dtype="Int64")})
    y = pd.Series([0, 0, 1, 1])
    step = encoder(smoothing=0).fit(X, y)
    result = step.transform(pd.DataFrame({"route": [100.0, 200.0]}))
    assert result.iloc[:, 0].tolist() == [0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_by": "route"},
        {"group_by": []},
        {"group_by": ["route", "route"]},
        {"group_by": [2]},
        {"smoothing": -1},
        {"smoothing": np.inf},
        {"smoothing": np.nan},
        {"smoothing": True},
        {"n_splits": 1},
        {"n_splits": 2.5},
        {"n_splits": True},
        {"seed": -1},
        {"seed": True},
        {"min_group_size": 0},
        {"output_name": ""},
    ],
)
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        encoder(**kwargs)


@pytest.mark.parametrize(
    "kind", ["none", "misaligned", "nonbinary", "missing", "wrong_length", "matrix"]
)
def test_bad_targets_are_rejected(kind):
    X, y = data()
    variants = {
        "none": None,
        "misaligned": y.iloc[::-1],
        "nonbinary": y.replace(1, 2),
        "missing": y.astype(float).mask(y.index == y.index[0]),
        "wrong_length": y.iloc[:2],
        "matrix": y.to_numpy()[:, None],
    }
    with pytest.raises(ValueError):
        encoder().fit_transform(X, variants[kind])


def test_array_targets_work_positionally():
    X, y = data()
    assert_frame_equal(encoder().fit_transform(X, y), encoder().fit_transform(X, y.to_numpy()))


def test_empty_single_class_and_too_small_cases():
    X, y = data()
    step = encoder()
    with pytest.raises(ValueError, match="n_splits"):
        step.fit_transform(X.iloc[:2], y.iloc[:2])
    with pytest.raises(ValueError, match="non-empty"):
        step.fit_transform(X.iloc[:0], y.iloc[:0])
    assert np.allclose(step.fit_transform(X, y * 0), 0)
    assert step.transform(X.iloc[:0]).shape == (0, 1)


def test_missing_columns_duplicate_index_and_unfitted_errors():
    X, y = data()
    with pytest.raises(RuntimeError, match="not fitted"):
        encoder().transform(X)
    with pytest.raises(KeyError, match="needs columns"):
        encoder().fit(X.drop(columns="route"), y)
    bad = X.copy()
    bad.index = [1] * len(X)
    with pytest.raises(ValueError, match="unique"):
        encoder().fit(bad, y)


def test_chunk_shuffle_and_joblib_parity(tmp_path):
    X, y = data()
    step = encoder()
    step.fit_transform(X, y)
    whole = step.transform(X)
    chunks = pd.concat([step.transform(X.iloc[i : i + 1]) for i in range(len(X))])
    assert_frame_equal(chunks, whole)
    assert_frame_equal(step.transform(X.sample(frac=1, random_state=7)).reindex(X.index), whole)
    path = tmp_path / "encoder.joblib"
    joblib.dump(step, path)
    assert_frame_equal(joblib.load(path).transform(X), whole)


class SpyInner:
    max_rounds = 100

    def fit_with_validation(self, X, y, X_val, y_val):
        self.train, self.val = X.copy(), X_val.copy()
        return 7

    def refit(self, X, y, n_rounds):
        self.train = X.copy()
        self.rounds = n_rounds

    def learning_curve(self, X, y, X_val, y_val, n_rounds, metric):
        self.train, self.val = X.copy(), X_val.copy()
        return pd.DataFrame({"round": [1], "metric": [metric]})

    def predict_proba(self, X):
        self.seen = X.copy()
        return X["te__route"].to_numpy()

    def predict_margin(self, X):
        self.seen = X.copy()
        return X["te__route"].to_numpy()

    def explain_margin(self, X, *, tolerance=None):
        self.seen = X.copy()
        return X

    def estimator_feature_names(self):
        return list(self.train.columns)

    def save(self, path):
        path.write_text("unit-test model")


def test_wrapper_training_crossfit_and_validation_frozen_mapping():
    X, y = data()
    val = pd.DataFrame({"route": ["A", "NEW"], "v": [999.0, 999.0]}, index=[101, 102])
    adapter = TransformingAdapter(SpyInner(), [SPEC])
    assert adapter.fit_with_validation(X, y, val, pd.Series([0, 1], index=val.index)) == 7
    expected_train = encoder().fit_transform(X, y)
    assert_series_equal(adapter.inner.train["te__route"], expected_train["te__route"])
    expected_val = encoder().fit(X, y).transform(val)
    assert_series_equal(adapter.inner.val["te__route"], expected_val["te__route"])
    other = TransformingAdapter(SpyInner(), [SPEC])
    other.fit_with_validation(X, y, val, pd.Series([1, 0], index=val.index))
    assert_frame_equal(other.inner.train, adapter.inner.train)
    assert_frame_equal(other.inner.val, adapter.inner.val)


def test_refit_uses_crossfit_but_prediction_uses_full_mapping():
    X, y = data()
    adapter = TransformingAdapter(SpyInner(), [SPEC])
    adapter.refit(X, y, 12)
    assert_series_equal(
        adapter.inner.train["te__route"], encoder().fit_transform(X, y)["te__route"]
    )
    assert np.allclose(adapter.predict_proba(X), encoder().fit(X, y).transform(X).iloc[:, 0])
    assert adapter.inner.rounds == 12
    assert adapter.estimator_feature_names() == ["route", "v", "te__route"]


def test_learning_curve_crossfits_training_inputs():
    X, y = data()
    inner = SpyInner()
    adapter = TransformingAdapter(inner, [SPEC])
    adapter.learning_curve(X, y, X, y, 10, "auc")
    assert_series_equal(inner.train["te__route"], encoder().fit_transform(X, y)["te__route"])
    assert_series_equal(inner.val["te__route"], encoder().fit(X, y).transform(X)["te__route"])


def test_mixed_supervised_and_unsupervised_steps_preserve_legacy_behavior():
    X, y = data()
    group = {"step": "group_statistics", "group_by": ["route"], "columns": ["v"], "stats": ["mean"]}
    power = {"step": "power_transform", "columns": ["v"]}
    transformer = FittedFeatureTransformer([SPEC, group, power])
    train = transformer.fit_transform(X, y)
    legacy = FittedFeatureTransformer([group, power]).fit(X).transform(X)
    assert_frame_equal(train[legacy.columns], legacy)
    assert list(train.columns) == transformer.output_columns(list(X.columns))


def test_wrapper_save_load_prediction_margin_and_explanation(tmp_path):
    X, y = data()
    adapter = TransformingAdapter(SpyInner(), [SPEC])
    adapter.refit(X, y, 7)
    expected = adapter.predict_proba(X)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    loaded = TransformingAdapter.load(SpyInner(), [SPEC], path)
    assert np.allclose(loaded.predict_proba(X), expected)
    assert np.allclose(loaded.predict_margin(X), expected)
    assert_frame_equal(loaded.explain_margin(X), loaded.transformer.transform(X))


def test_registration_collisions_and_label_requirement():
    validate_specs([SPEC])
    with pytest.raises(TypeError):
        TargetEncodingStep(group_by=["route"], unknown=1)
    X, y = data()
    with pytest.raises(ValueError, match="requires training labels"):
        FittedFeatureTransformer([SPEC]).fit(X)
    with pytest.raises(ValueError, match="requires training labels"):
        FittedFeatureTransformer([SPEC]).fit_transform(X)
    clash = X.assign(te__route=0.0)
    with pytest.raises(ValueError, match="duplicate"):
        FittedFeatureTransformer([SPEC]).fit_transform(clash, y)
    with pytest.raises(ValueError, match="duplicate"):
        FittedFeatureTransformer([SPEC, SPEC]).fit_transform(X, y)


def test_real_sklearn_estimator_training_and_saved_transformer(tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    class TreeInner:
        max_rounds = 5

        def refit(self, X, y, n_rounds):
            self.model = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X, y)

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, 1]

        def estimator_feature_names(self):
            return list(self.model.feature_names_in_)

        def save(self, path):
            joblib.dump(self.model, path)

    X = pd.DataFrame({"route": np.tile([100, 200, 300], 12), "age": np.arange(36)})
    y = pd.Series(np.tile([0, 1, 1, 0, 1, 0], 6), index=X.index)
    adapter = TransformingAdapter(TreeInner(), [SPEC])
    adapter.refit(X, y, 3)
    p = adapter.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    path = tmp_path / "tree.joblib"
    adapter.save(path)
    inner = TreeInner()
    inner.model = joblib.load(path)
    loaded = TransformingAdapter.load(inner, [SPEC], path)
    assert np.allclose(loaded.predict_proba(X), p)
