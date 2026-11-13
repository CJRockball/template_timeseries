import copy

import numpy as np
import pandas as pd
import pytest

from ml_template.preprocessing.contracts import validate_estimator_frame, validate_nn_arrays
from ml_template.preprocessing.nn_config import NNPreprocessorConfig, load_nn_preprocessor_config
from ml_template.preprocessing.nn_preprocessor import NNPreprocessor


def mixed():
    return pd.DataFrame(
        {
            "x": pd.Series([1, 2, None, 10], dtype="Float64"),
            "cat": pd.Series(["A", None, "missing", "__UNKNOWN__"], dtype="string"),
        }
    )


def prep(**kwargs):
    return NNPreprocessor(NNPreprocessorConfig(numeric=["x"], categorical=["cat"], **kwargs))


@pytest.mark.parametrize("strategy", ["mean", "median", "constant"])
@pytest.mark.parametrize("scaling", ["standard", "none"])
def test_numeric_statistics_on_observed_values(strategy, scaling):
    X = mixed()
    p = prep(numeric_strategy=strategy, scaling=scaling, constant_fill=7).fit(X)
    stats = p.state_dict()["numeric_state"]["x"]
    observed = np.array([1.0, 2.0, 10.0])
    fill = {"mean": observed.mean(), "median": 2.0, "constant": 7.0}[strategy]
    assert stats["fill"] == pytest.approx(fill)
    assert stats["center"] == pytest.approx(observed.mean() if scaling == "standard" else 0)
    assert stats["scale"] == pytest.approx(observed.std() if scaling == "standard" else 1)
    out = p.transform(X)
    assert out.x.iloc[2] == pytest.approx((fill - stats["center"]) / stats["scale"])
    assert out.x.dtype == np.float32
    assert out.x__is_missing.tolist() == [0.0, 0.0, 1.0, 0.0]
    pd.testing.assert_frame_equal(X, mixed())


def test_mean_imputation_is_zero_in_standardised_space():
    p = prep(numeric_strategy="mean").fit(mixed())
    assert p.transform(mixed()).x.iloc[2] == pytest.approx(0)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64"])
def test_all_missing_numeric_and_constant(dtype):
    X = pd.DataFrame({"x": pd.Series([None, None], dtype=dtype), "k": [5.0, 5.0]})
    p = NNPreprocessor({"numeric": ["x", "k"], "all_missing_fill": -3}).fit(X)
    out = p.transform(X)
    assert out.x.tolist() == [-3.0, -3.0]
    assert out.k.tolist() == [0.0, 0.0]
    assert p.state_dict()["numeric_state"]["k"]["scale"] == 1
    assert out.x__is_missing.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
def test_branch_shapes_flags_dtypes_and_contiguity(kind):
    config = {
        "numeric": ["x"] if kind != "categorical" else [],
        "categorical": ["cat"] if kind != "numeric" else [],
    }
    columns = config["numeric"] + config["categorical"]
    X = mixed()[columns]
    p = NNPreprocessor(config).fit(X)
    a = p.transform_arrays(X)
    assert a.numeric.shape == (4, len(config["numeric"]) + len(columns))
    assert a.categorical.shape == (4, len(config["categorical"]))
    assert a.numeric.dtype == np.float32
    assert a.categorical.dtype == np.int64
    assert a.numeric.flags.c_contiguous and a.categorical.flags.c_contiguous
    empty = p.transform_arrays(X.iloc[:0])
    assert empty.numeric.shape[0] == empty.categorical.shape[0] == 0


def test_missing_unknown_and_literal_sentinels_are_distinct():
    p = prep().fit(mixed())
    X = pd.DataFrame({"x": [2.0, 2.0, 2.0, 2.0], "cat": [None, "new", "missing", "__UNKNOWN__"]})
    out = p.transform(X)
    assert out.cat.tolist()[:2] == [0, 1]
    assert len(set(out.cat)) == 4
    assert out.cat__is_missing.tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_categorical_string_dtypes(dtype):
    X = pd.DataFrame({"c": pd.Series(["A", None, "B"], dtype=dtype)})
    p = NNPreprocessor({"categorical": ["c"]}).fit(X)
    assert p.transform(X).c.iloc[1] == 0
    assert p.cardinalities == (4,)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "category"])
def test_integer_categorical_missing_and_float_equivalence(dtype):
    X = pd.DataFrame({"c": pd.Series([1, None, 2], dtype=dtype)})
    p = NNPreprocessor({"categorical": ["c"]}).fit(X)
    result = p.transform(pd.DataFrame({"c": [1.0, np.nan, 3.0]}))
    assert result.c.tolist() == [2, 0, 1]


def test_type_sensitive_categories_and_json_unicode_roundtrip(tmp_path):
    values = [1, "1", True, 1.5, "☀", "nan", "", None]
    X = pd.DataFrame({"c": pd.Series(values, dtype=object)})
    p = NNPreprocessor({"categorical": ["c"]}).fit(X)
    assert len(set(p.transform(X).c)) == len(values)
    path = tmp_path / "pre.json"
    p.save(path)
    pd.testing.assert_frame_equal(p.transform(X), NNPreprocessor.load(path).transform(X))


def test_unused_full_frame_categories_are_not_learned():
    X = pd.DataFrame({"c": pd.Categorical(["A", None], categories=["A", "B", "HOLDOUT"])})
    p = NNPreprocessor({"categorical": ["c"]}).fit(X)
    assert p.cardinalities == (3,)
    assert p.transform(pd.DataFrame({"c": ["HOLDOUT"]})).c.iloc[0] == 1


def test_unknown_error_policy():
    p = prep(unknown_policy="error").fit(mixed())
    with pytest.raises(ValueError, match="unknown category"):
        p.transform(pd.DataFrame({"x": [1.0], "cat": ["new"]}))


def test_new_missingness_does_not_change_feature_schema():
    train = pd.DataFrame({"x": [1.0, 2.0], "c": ["A", "B"]})
    p = NNPreprocessor({"numeric": ["x"], "categorical": ["c"]}).fit(train)
    test = pd.DataFrame({"x": [np.nan], "c": [None]})
    assert p.transform(test).columns.tolist() == p.transform(train).columns.tolist()
    assert p.transform(test).iloc[0].tolist()[-2:] == [1.0, 1.0]


def test_all_missing_category_has_two_reserved_ids():
    X = pd.DataFrame({"c": [None, None]})
    p = NNPreprocessor({"categorical": ["c"]}).fit(X)
    assert p.cardinalities == (2,)
    assert p.transform(pd.DataFrame({"c": [None, "A"]})).c.tolist() == [0, 1]


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinities_are_rejected_in_numeric_and_categorical(value):
    for role in ["numeric", "categorical"]:
        p = NNPreprocessor({role: ["x"]})
        with pytest.raises(ValueError, match="infinite|nonfinite"):
            p.fit(pd.DataFrame({"x": [value]}))


def test_numeric_float32_overflow_and_string_input_rejected():
    p = NNPreprocessor({"numeric": ["x"], "scaling": "none"}).fit(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="float32"):
        p.transform(pd.DataFrame({"x": [1e40]}))
    with pytest.raises(ValueError, match="real numeric"):
        p.transform(pd.DataFrame({"x": ["2"]}))


def test_fit_overflow_rejected():
    with pytest.raises(ValueError, match="statistics"):
        NNPreprocessor({"numeric": ["x"]}).fit(pd.DataFrame({"x": [1e308, 1e308]}))


def test_frozen_state_independent_of_validation_and_refit():
    p = prep().fit(mixed())
    before = p.state_dict()
    p.transform(pd.DataFrame({"x": [1000.0], "cat": ["new"]}))
    assert p.state_dict() == before
    copied = p.state_dict()
    copied["numeric_state"]["x"]["fill"] = -999
    assert p.state_dict() == before
    p.fit(pd.DataFrame({"x": [20.0, 40.0], "cat": ["Z", "Z"]}))
    assert p.state_dict()["fit_rows"] == 2
    assert p.cardinalities == (3,)
    assert p.state_dict() != before


def test_independent_fold_and_final_states():
    X = mixed()
    a, b = prep().fit(X.iloc[:2]), prep().fit(X.iloc[2:])
    a_state = a.state_dict()
    assert a_state != b.state_dict()
    final = prep().fit(X)
    assert a.state_dict() == a_state
    assert final.state_dict()["fit_rows"] == 4


def test_chunk_shuffle_singleton_and_column_order_parity():
    p = prep().fit(mixed())
    X = mixed()
    expected = p.transform(X)
    pd.testing.assert_frame_equal(
        expected, pd.concat([p.transform(X.iloc[:1]), p.transform(X.iloc[1:])])
    )
    pd.testing.assert_frame_equal(expected, p.transform(X.iloc[::-1]).iloc[::-1])
    pd.testing.assert_frame_equal(expected.iloc[:1], p.transform(X.iloc[:1]))
    pd.testing.assert_frame_equal(expected, p.transform(X[["cat", "x"]]))


def test_preserves_duplicate_and_named_indexes():
    X = mixed()
    X.index = pd.Index([7, 7, 2, 9], name="entity")
    assert prep().fit_transform(X).index.equals(X.index)


@pytest.mark.parametrize("change", ["extra", "missing", "duplicate", "empty_fit"])
def test_input_schema_failures(change):
    p = prep().fit(mixed())
    X = mixed()
    if change == "extra":
        X["target"] = 0
    elif change == "missing":
        X = X.drop(columns="cat")
    elif change == "duplicate":
        X = pd.concat([X, X[["x"]]], axis=1)
    else:
        X = X.iloc[:0]
    with pytest.raises(ValueError):
        p.fit(X) if change == "empty_fit" else p.transform(X)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"numeric": ["x"], "categorical": ["x"]},
        {"numeric": ["x", "x__is_missing"]},
        {"numeric": "x"},
        {"numeric": ["x"], "numeric_strategy": "mode"},
        {"numeric": ["x"], "scaling": "robust"},
        {"numeric": ["x"], "unknown_policy": "guess"},
        {"numeric": ["x"], "all_missing_fill": float("nan")},
        {"numeric": ["x"], "constant_fill": True},
        {"numeric": ["x"], "device": "cuda"},
    ],
)
def test_strict_config(raw):
    with pytest.raises(ValueError):
        NNPreprocessorConfig.from_dict(raw)


def test_yaml_config(tmp_path):
    path = tmp_path / "pre.yaml"
    path.write_text("numeric: [x]\ncategorical: []\nnumeric_strategy: mean\n")
    config = load_nn_preprocessor_config(path)
    assert config.numeric == ("x",)
    assert config.numeric_strategy == "mean"


def test_save_load_no_overwrite_and_unfitted(tmp_path):
    p = prep()
    path = tmp_path / "state.json"
    with pytest.raises(RuntimeError):
        p.save(path)
    assert not path.exists()
    with pytest.raises(RuntimeError):
        p.transform(mixed())
    p.fit(mixed()).save(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        p.save(path)
    assert path.read_bytes() == before
    loaded = NNPreprocessor.load(path)
    pd.testing.assert_frame_equal(p.transform(mixed()), loaded.transform(mixed()))


@pytest.mark.parametrize(
    "change",
    ["version", "extra", "scale", "nonfinite", "vocabulary", "duplicate_token", "type", "fit_rows"],
)
def test_reject_corrupt_checkpoint(change):
    state = copy.deepcopy(prep().fit(mixed()).state_dict())
    if change == "version":
        state["schema_version"] = 2
    elif change == "extra":
        state["unexpected"] = 1
    elif change == "scale":
        state["numeric_state"]["x"]["scale"] = 0
    elif change == "nonfinite":
        state["numeric_state"]["x"]["fill"] = float("inf")
    elif change == "vocabulary":
        state["vocabularies"]["cat"] = ["not-json"]
    elif change == "duplicate_token":
        state["vocabularies"]["cat"] *= 2
    elif change == "type":
        state["numeric_state"]["x"]["n_observed"] = True
    else:
        state["fit_rows"] = 0
    with pytest.raises(ValueError):
        NNPreprocessor.from_state_dict(state)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        NNPreprocessor.load(path)


@pytest.mark.parametrize(
    "bad", ["float_dtype", "int_dtype", "negative", "range", "nan", "rows", "shape", "size"]
)
def test_array_contract_failures(bad):
    n = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.int64)
    sizes = (2,)
    if bad == "float_dtype":
        n = n.astype(np.float64)
    elif bad == "int_dtype":
        c = c.astype(np.int32)
    elif bad == "negative":
        c[0, 0] = -1
    elif bad == "range":
        c[0, 0] = 2
    elif bad == "nan":
        n[0, 0] = np.nan
    elif bad == "rows":
        c = c[:1]
    elif bad == "shape":
        n = n[:, 0]
    else:
        sizes = (True,)
    with pytest.raises(ValueError):
        validate_nn_arrays(n, c, sizes)


def test_original_frame_validator_behaviour_unchanged():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    assert validate_estimator_frame(X) is X
    with pytest.raises(ValueError, match="missing"):
        validate_estimator_frame(pd.DataFrame({"x": [np.nan]}))
    with pytest.raises(ValueError, match="numeric"):
        validate_estimator_frame(pd.DataFrame({"x": ["A"]}))
    with pytest.raises(ValueError, match="infinite"):
        validate_estimator_frame(pd.DataFrame({"x": [float("inf")]}))
    Y = pd.DataFrame({"x": [np.nan]})
    assert validate_estimator_frame(Y, allow_nan=True) is Y
