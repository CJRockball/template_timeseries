import numpy as np
import pandas as pd
import pytest

from ml_template.preprocessing.contracts import validate_estimator_frame
from ml_template.scoring.input_contract import coerce_frame, frame_schema, validate_batch
from ml_template.scoring.pre_gold_schema import build_pre_gold_contract, get_preprocessor_schema


def source():
    return pd.DataFrame(
        {
            "arrival": pd.array([5, None, 9], dtype="Int64"),
            "departure": pd.array([2, 3, 4], dtype="Int64"),
            "kind": pd.Categorical(["a", "b", "a"]),
        }
    )


def contract():
    raw = source()
    features = raw.assign(delta=raw.arrival - raw.departure)
    return build_pre_gold_contract(raw, features, key_col="id", key_kind="int", input_spec={})


def test_engineered_nulls_accepted_without_imputation():
    raw = source()
    features = raw.assign(delta=raw.arrival - raw.departure)
    schema = get_preprocessor_schema(contract())
    out = coerce_frame(features, schema, enforce_rates=True)
    assert out.delta.isna().tolist() == [False, True, False]
    assert out.delta.dropna().tolist() == [3, 5]


def test_null_permission_not_inferred_from_training_observations():
    frame = pd.DataFrame({"x": pd.array([1, 2], dtype="Int64")})
    schema = frame_schema(frame, nullable=True)
    assert schema[0]["observed_missing_rate"] == 0
    future = pd.DataFrame({"x": pd.array([None, None], dtype="Int64")})
    assert coerce_frame(future, schema, enforce_rates=True).x.isna().all()


def test_numpy_integer_schema_promoted_for_future_nulls():
    schema = frame_schema(pd.DataFrame({"x": np.array([1, 2], dtype="int16")}), nullable=True)
    assert schema[0]["dtype"] == "Int16"
    got = coerce_frame(pd.DataFrame({"x": pd.array([None], dtype="Int16")}), schema)
    assert got.x.isna().all()


def test_nullable_boolean_preserved():
    schema = frame_schema(pd.DataFrame({"flag": [True, False]}), nullable=True)
    got = coerce_frame(pd.DataFrame({"flag": pd.array([None, True], dtype="boolean")}), schema)
    assert got.flag.isna().tolist() == [True, False]


def test_legacy_strict_default_unchanged():
    with pytest.raises(ValueError, match="missing"):
        frame_schema(source())


def test_legacy_rate_limit_unchanged():
    with pytest.raises(ValueError, match="rate"):
        frame_schema(source(), missingness={"default_max_rate": 0.1})


def test_explicit_nonnullable_stays_strict():
    schema = frame_schema(pd.DataFrame({"x": [1.0]}), nullable=False)
    with pytest.raises(ValueError, match="missing"):
        coerce_frame(pd.DataFrame({"x": [np.nan]}), schema)


def test_schema_modes_cannot_be_mixed():
    with pytest.raises(ValueError, match="OR"):
        frame_schema(source(), nullable=True, missingness={})


@pytest.mark.parametrize("nullable", ["true", 1, []])
def test_nonboolean_nullability_rejected(nullable):
    with pytest.raises(ValueError):
        frame_schema(source(), nullable=nullable)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_values_still_rejected(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        frame_schema(pd.DataFrame({"x": [bad]}), nullable=True)


def test_key_nulls_still_rejected():
    frame = source()
    frame.insert(0, "id", pd.array([1, None, 3], dtype="Int64"))
    with pytest.raises(ValueError, match="keys"):
        validate_batch(frame, contract(), enforce_rates=True)


def test_duplicate_key_still_rejected():
    frame = source()
    frame.insert(0, "id", [1, 1, 3])
    with pytest.raises(ValueError, match="keys"):
        validate_batch(frame, contract(), enforce_rates=True)


def test_unseen_category_still_rejected():
    schema = frame_schema(source(), nullable=True)
    with pytest.raises(ValueError, match="unseen"):
        coerce_frame(source().assign(kind="unexpected"), schema)


def test_preprocessor_schema_alias_is_explicit_and_consistent():
    frozen = contract()
    assert frozen["model_schema_stage"] == "pre_gold"
    assert frozen["model_schema"] == frozen["preprocessor_input_schema"]
    frozen["model_schema"] = []
    with pytest.raises(ValueError, match="alias"):
        get_preprocessor_schema(frozen)


@pytest.mark.parametrize("version", [1, 2])
def test_old_contract_resolver(version):
    assert get_preprocessor_schema({"schema_version": version, "model_schema": ["sentinel"]}) == [
        "sentinel"
    ]


def test_no_legacy_override_silently_ignored():
    with pytest.raises(ValueError, match="Remove"):
        build_pre_gold_contract(
            source(),
            source(),
            key_col="id",
            key_kind="int",
            input_spec={},
            legacy_model_limits={"arrival": 1.0},
        )


def test_null_rates_are_observations_not_imputation():
    frozen = contract()
    delta = next(item for item in frozen["preprocessor_input_schema"] if item["name"] == "delta")
    assert delta["observed_missing_rate"] == pytest.approx(1 / 3)
    assert delta["nullable"] is True
    assert "max_missing_rate" not in delta


def test_post_gold_validator_blocks_nan_without_modifying_frame():
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="post-Gold missing"):
        validate_estimator_frame(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_post_gold_validator_accepts_finite_numeric_inputs():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    assert validate_estimator_frame(frame) is frame


def test_explicit_tree_capability_can_preserve_nan():
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    assert validate_estimator_frame(frame, allow_nan=True) is frame


def test_post_gold_validator_rejects_infinity_even_if_nan_allowed():
    with pytest.raises(ValueError, match="infinite"):
        validate_estimator_frame(pd.DataFrame({"x": [np.inf]}), allow_nan=True)


def test_post_gold_validator_rejects_unencoded_categories():
    with pytest.raises(ValueError, match="numeric"):
        validate_estimator_frame(pd.DataFrame({"x": ["a"]}))
