import numpy as np
import pandas as pd
import pytest

from ml_template.data.canonicalize import canonicalize_frame
from ml_template.scoring.export_predictions import export_frame
from ml_template.scoring.input_contract import coerce_frame, frame_schema, validate_batch


def policy(limit=0.5):
    return {"default_max_rate": 0.0, "max_rate_by_column": {"delay": limit}}


def sample():
    return pd.DataFrame(
        {
            "delay": pd.array([1, None, 3, 4], dtype="Int64"),
            "kind": pd.Categorical(["a", "b", "a", "b"]),
        }
    )


def test_unspecified_nulls_rejected():
    with pytest.raises(ValueError, match="missing"):
        frame_schema(sample())


def test_nullable_int_and_category_roundtrip():
    frame = sample()
    schema = frame_schema(frame, missingness=policy())
    pd.testing.assert_frame_equal(coerce_frame(frame, schema, enforce_rates=True), frame)


def test_high_batch_rate_rejected_but_singleton_replay_works():
    frame = sample()
    schema = frame_schema(frame, missingness=policy())
    singleton = frame.iloc[[1]]
    assert coerce_frame(singleton, schema).delay.isna().all()
    with pytest.raises(ValueError, match="rate"):
        coerce_frame(singleton, schema, enforce_rates=True)


@pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt64"])
def test_nullable_integer_dtypes(dtype):
    frame = pd.DataFrame({"delay": pd.array([1, None], dtype=dtype)})
    pd.testing.assert_frame_equal(
        coerce_frame(frame, frame_schema(frame, missingness=policy())), frame
    )


def test_nullable_float_and_boolean():
    frame = pd.DataFrame(
        {
            "delay": pd.array([1.5, None], dtype="Float64"),
            "flag": pd.array([True, None], dtype="boolean"),
        }
    )
    rules = {"default_max_rate": 0.5}
    pd.testing.assert_frame_equal(
        coerce_frame(frame, frame_schema(frame, missingness=rules)), frame
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.assign(kind="unseen"),
        lambda f: f.assign(delay="2"),
        lambda f: f.assign(delay=True),
        lambda f: f.assign(delay=1.5),
        lambda f: f.assign(delay=np.inf),
        lambda f: f.assign(extra=1),
        lambda f: f.drop(columns="kind"),
        lambda f: f.iloc[:0],
    ],
)
def test_strict_rejections_preserved(change):
    frame = sample()
    schema = frame_schema(frame, missingness=policy())
    with pytest.raises(ValueError):
        coerce_frame(change(frame), schema)


def test_nullable_overflow_rejected():
    schema = frame_schema(
        pd.DataFrame({"delay": pd.array([1, None], dtype="Int8")}), missingness=policy()
    )
    with pytest.raises(ValueError, match="range"):
        coerce_frame(pd.DataFrame({"delay": [128]}), schema)


def test_large_integer_not_rounded():
    frame = pd.DataFrame({"delay": pd.array([2**53 + 1, None], dtype="Int64")})
    got = coerce_frame(frame, frame_schema(frame, missingness=policy()))
    assert got.delay.iloc[0] == 2**53 + 1


def test_keys_stay_nonnullable_and_unique():
    frame = sample()
    schema = {
        "key_col": "id",
        "key_kind": "int",
        "source_schema": frame_schema(frame, missingness=policy()),
    }
    frame.insert(0, "id", pd.array([9, 8, 7, 6], dtype="Int64"))
    assert validate_batch(frame, schema, enforce_rates=True).index.tolist() == [9, 8, 7, 6]
    with pytest.raises(ValueError):
        validate_batch(frame.assign(id=[1, 1, 2, 3]), schema)
    frame.loc[0, "id"] = pd.NA
    with pytest.raises(ValueError):
        validate_batch(frame, schema)


def spec():
    return {
        "canonical_names": {"ID": "id", "Delay": "delay", "Label": "label"},
        "canonical_dtypes": {"id": "Int64", "delay": "Int16", "label": "int8"},
        "value_mappings": {"label": {"False": 0, "True": 1}},
        "allowed_values": {"label": [0, 1]},
    }


def test_shared_labelled_and_unlabelled_canonicalisation():
    raw = pd.DataFrame({"ID": ["1", "2"], "Delay": ["3", None], "Label": ["False", "True"]})
    labelled = canonicalize_frame(raw, spec())
    assert labelled.label.tolist() == [0, 1]
    unlabelled = canonicalize_frame(raw.drop(columns="Label"), spec(), columns=["id", "delay"])
    pd.testing.assert_frame_equal(labelled[["id", "delay"]], unlabelled)


@pytest.mark.parametrize("bad", ["128.5", "40000", "inf", "oops"])
def test_canonical_integer_rejects_invalid(bad):
    with pytest.raises((ValueError, TypeError)):
        canonicalize_frame(
            pd.DataFrame({"ID": ["1"], "Delay": [bad]}), spec(), columns=["id", "delay"]
        )


def test_unknown_target_not_silently_null():
    with pytest.raises(ValueError, match="unmapped"):
        canonicalize_frame(
            pd.DataFrame({"ID": ["1"], "Delay": ["2"], "Label": ["unknown"]}), spec()
        )


def test_export_is_keyed_and_ordered():
    predictions = pd.DataFrame({"id": [2, 1], "p_release": [0.2, 0.8]})
    reference = pd.DataFrame({"id": [1, 2], "answer": [0, 0]})
    out = export_frame(predictions, reference, key="id", score="p_release", output_column="answer")
    assert out.columns.tolist() == ["id", "answer"]
    assert out.answer.tolist() == [0.8, 0.2]


@pytest.mark.parametrize(
    "scores", [[np.nan, 0.1], [np.inf, 0.1], [-0.1, 0.1], [1.1, 0.2], [True, False]]
)
def test_export_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        export_frame(
            pd.DataFrame({"id": [1, 2], "p": scores}),
            pd.DataFrame({"id": [1, 2]}),
            key="id",
            score="p",
            output_column="answer",
        )


def test_export_rejects_key_mismatch():
    with pytest.raises(ValueError, match="coverage"):
        export_frame(
            pd.DataFrame({"id": [1, 3], "p": [0.1, 0.2]}),
            pd.DataFrame({"id": [1, 2]}),
            key="id",
            score="p",
            output_column="answer",
        )


def test_frozen_typed_mapping_survives_json_roundtrip():
    import json

    frozen = {
        "canonical_names": {"code": "code"},
        "canonical_dtypes": {"code": "int8"},
        "value_mappings": {"code": [{"source": 10, "value": 0}, {"source": 20, "value": 1}]},
    }
    frozen = json.loads(json.dumps(frozen))
    got = canonicalize_frame(pd.DataFrame({"code": [10, 20]}), frozen)
    assert got.code.tolist() == [0, 1]


def test_training_lineage_is_explicitly_preserved():
    raw = pd.DataFrame(
        {
            "ID": ["1"],
            "Delay": ["2"],
            "Label": ["True"],
            "dataset_version_id": ["v1"],
            "source_row_number": [1],
        }
    )
    with pytest.raises(ValueError, match="undeclared"):
        canonicalize_frame(raw, spec())
    got = canonicalize_frame(raw, spec(), passthrough=("dataset_version_id", "source_row_number"))
    assert got.dataset_version_id.tolist() == ["v1"]
    assert got.source_row_number.tolist() == [1]
