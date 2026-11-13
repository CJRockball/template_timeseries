"""Pure batch/export identities, configuration and pinned-runtime orchestration contracts."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml_template.scoring.nn_batch_contract import (
    export_frame,
    key_fingerprint,
    validate_export_mapping,
    validate_predictions,
    verify_batch,
)
from ml_template.scoring.nn_batch_pipeline import BatchRequest, path_for, reference_keys


def world():
    rel = "rel_" + "0" * 24
    batch = "score_test"
    info = {
        "task": "regression",
        "prediction_kind": "continuous",
        "model_type": "nn",
        "parent_run_id": "run_test",
        "calibration_id": None,
        "release_variant": "raw",
        "target_unit": "MPa",
        "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
    }
    frame = pd.DataFrame(
        {
            "id": [3, 1, 2],
            "y_pred_raw": [-1.0, 2.0, 3.0],
            "y_pred_release": [-1.0, 2.0, 3.0],
            "release_id": rel,
            "scoring_batch_id": batch,
        }
    )
    summary = {
        "release_id": rel,
        "scoring_batch_id": batch,
        "parent_run_id": "run_test",
        "calibration_id": None,
        "release_variant": "raw",
        "prediction_columns": info["prediction_columns"],
        "task": "regression",
        "prediction_kind": "continuous",
        "labels_loaded": False,
        "n_flagged": None,
        "n_rows": 3,
    }
    return (info, frame, summary, rel, batch)


def test_frozen_batch_and_raw_unit_fallback():
    info, frame, summary, rel, batch = world()
    validate_predictions(frame, summary, info, "id", rel, batch)
    assert info["target_unit"] == "MPa" and "target_unit" not in summary


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_run_id", "wrong"),
        ("release_id", "wrong"),
        ("calibration_id", "wrong"),
        ("release_variant", "recalibrated"),
        ("target_unit", "wrong"),
        ("task", "binary_classification"),
        ("prediction_kind", "probability"),
        ("labels_loaded", True),
        ("n_flagged", 1),
        ("n_rows", 2),
    ],
)
def test_bad_batch_summary(field, value):
    info, frame, summary, rel, batch = world()
    summary[field] = value
    with pytest.raises(ValueError):
        validate_predictions(frame, summary, info, "id", rel, batch)


@pytest.mark.parametrize(
    "kind", ["duplicate", "null", "extra", "nan", "boolean", "release", "batch", "uncorrected"]
)
def test_bad_scored_predictions(kind):
    info, frame, summary, rel, batch = world()
    if kind == "duplicate":
        frame.loc[0, "id"] = 1
    elif kind == "null":
        frame.loc[0, "id"] = np.nan
    elif kind == "extra":
        frame["y_true"] = 1.0
    elif kind == "nan":
        frame.loc[0, "y_pred_raw"] = np.nan
    elif kind == "boolean":
        frame["y_pred_raw"] = True
    elif kind == "release":
        frame.loc[0, "release_id"] = "wrong"
    elif kind == "batch":
        frame.loc[0, "scoring_batch_id"] = "wrong"
    else:
        frame.loc[0, "y_pred_release"] += 1
    with pytest.raises(ValueError):
        validate_predictions(frame, summary, info, "id", rel, batch)


def test_export_preserves_full_reference_order_and_negative_values():
    _, frame, _, _, _ = world()
    ref = pd.DataFrame({"id": [2, 3, 1]})
    before = frame.copy(deep=True)
    out = export_frame(
        frame,
        ref,
        key="id",
        columns={"y_pred_raw": "raw", "y_pred_release": "prediction"},
        allowed=["y_pred_raw", "y_pred_release"],
    )
    assert out.id.tolist() == [2, 3, 1] and out.prediction.tolist() == [3.0, -1.0, 2.0]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "columns",
    [
        {},
        {"y_true": "prediction"},
        {"y_pred_raw": "id"},
        {"y_pred_raw": "same", "y_pred_release": "same"},
        {"y_pred_release": ""},
    ],
)
def test_export_mapping_rejected(columns):
    with pytest.raises(ValueError):
        validate_export_mapping(columns, "id", ["y_pred_raw", "y_pred_release"])


@pytest.mark.parametrize(
    "keys", [[1, 2], [1, 1, 2], [1, None, 2], [1.0, 2.0, 3.0], ["1", "2", "3"]]
)
def test_no_intersection_or_type_coercion_on_export(keys):
    _, frame, _, _, _ = world()
    with pytest.raises(ValueError):
        export_frame(
            frame,
            pd.DataFrame({"id": keys}),
            key="id",
            columns={"y_pred_release": "prediction"},
            allowed=["y_pred_release"],
        )


def test_string_keys_preserve_leading_zeroes():
    p = pd.DataFrame({"key": ["01", "1"], "v": [-3.0, 4.0]})
    r = pd.DataFrame({"key": ["1", "01"]})
    out = export_frame(p, r, key="key", columns={"v": "prediction"}, allowed=["v"])
    assert out.key.tolist() == ["1", "01"] and out.prediction.tolist() == [4.0, -3.0]


def test_key_fingerprint_order_invariant_but_type_sensitive():
    assert key_fingerprint([1, 2]) == key_fingerprint([2, 1])
    assert key_fingerprint([1, 2]) != key_fingerprint(["1", "2"])


@pytest.mark.parametrize("keys", [[], [True, False], [1, "2"], [1, 1], [None, 1], [1.0, 2.0]])
def test_bad_key_fingerprints(keys):
    with pytest.raises(ValueError):
        key_fingerprint(keys)


def request(**extra):
    return BatchRequest(
        **{
            "release_dir": "release",
            "expected_release_id": "rel_" + "0" * 24,
            "input_path": "input.csv",
            "runtime_source_root": "pinned/src/ml_template",
            **extra,
        }
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"expected_release_id": "latest"},
        {"chunk_size": True},
        {"chunk_size": 0},
        {"timeout_s": 0},
        {"acknowledge_research_scoring": "true"},
        {"runtime_source_root": ""},
        {"evaluation_config": "new_eval.yaml"},
        {"train": True},
    ],
)
def test_bad_requests(extra):
    with pytest.raises(ValueError):
        request(**extra)


def test_safe_default_never_authorises_execute():
    assert request().acknowledge_research_scoring is False


def test_venv_python_symlink_path_preserved(tmp_path, monkeypatch):
    actual = tmp_path / "base_python"
    actual.write_text("fixture")
    link = tmp_path / "venv_python"
    link.symlink_to(actual)
    assert path_for(str(link), follow_symlinks=False) == link
    assert path_for(str(link)) == actual


def test_reference_reads_keys_only(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("id,prediction\n01,999\n1,888\n")
    out = reference_keys(path, "id", "str", ["prediction"])
    assert out.columns.tolist() == ["id"] and out.id.tolist() == ["01", "1"]


def test_reference_rejects_unexpected_labels(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("id,y_true\n1,99\n")
    with pytest.raises(ValueError):
        reference_keys(path, "id", "int", ["prediction"])


def make_batch(root):
    (root / "predictions.parquet").write_bytes(b"hash-only fixture")
    (root / "summary.json").write_text(json.dumps({"fixture": True}))
    entries = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in root.iterdir()
    }
    (root / "artifacts.json").write_text(json.dumps({"files": entries}))


def test_scoring_hashes_checked_before_parquet(tmp_path):
    make_batch(tmp_path)
    assert verify_batch(tmp_path) == {"fixture": True}
    (tmp_path / "predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify_batch(tmp_path)


def test_unlisted_batch_files_rejected(tmp_path):
    make_batch(tmp_path)
    (tmp_path / "extra").write_text("x")
    with pytest.raises(ValueError):
        verify_batch(tmp_path)
