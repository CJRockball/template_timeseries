"""Pure frozen NN evidence and read-only authorization contracts."""

import copy
import json
import sqlite3
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation import nn_evaluate as nn
from ml_template.tracking import nn_evaluation_contract as c


def evidence(method=None):
    d = {
        "schema_version": 3,
        "task": "regression",
        "prediction_kind": "continuous",
        "model_type": "nn",
        "kind": "final",
        "weighting": "equal",
        "fitted_features": False,
        "serialization": {
            "format": "torch",
            "weights_only": True,
            "checkpoint_version": 1,
            "preprocessing": "embedded",
        },
        "members": [
            {"fold_id": None, "model_path": "model.ubj", "transformer_path": None, "weight": 1.0}
        ],
        "files_sha256": {k: "0" * 64 for k in ["model.ubj", "config.json", "features.json"]},
        "training_rows": {"model.ubj": 20},
    }
    fp = c.digest(
        {
            "silver": "silver_test",
            "key": "id",
            "members": [json.dumps(["int", v]) for v in [10, 11, 12]],
        }
    )
    cfg = {
        "schema_version": 2,
        "task": "regression",
        "parent_run_id": "run_test",
        "predictor": {"kind": "final"},
        "primary_metric": "rmse",
        "expected_silver_dataset_version_id": "silver_test",
        "expected_split_id": "split_test",
        "expected_feature_set_id": None,
        "target_unit": "MPa",
        "release_variant": "raw",
        "calibration_id": None,
        "uncertainty": {"enabled": False},
        "criteria": [],
    }
    p = {
        "nn_evaluation_schema": 1,
        "predictor": d,
        "input_hashes": {},
        "code_hashes": {},
        "feature_preparation": {},
        "versions": {},
        "python": "fixture",
        "holdout_fingerprint": fp,
    }
    raw = np.array([-1.0, 0.0, 2.0])
    point = raw.copy()
    if method is not None:
        correction = {
            "method": method,
            "slope": 2.0 if method == "affine" else 1.0,
            "intercept": 0.0 if method == "none" else 1.0,
        }
        cfg.update(
            calibration_id="regcal_test",
            release_variant="raw" if method == "none" else "recalibrated",
            uncertainty={"enabled": True, "coverages": [0.8]},
        )
        state = {
            "schema_version": 1,
            "calibration_id": "regcal_test",
            "holdout_labels_loaded": False,
            "config": {
                "model_type": "nn",
                "parent_run_id": "run_test",
                "predictor_kind": "final",
                "target_unit": "MPa",
                "correction": method,
                "coverages": [0.8],
            },
            "binding": c.prediction_binding(p),
            "parameters": {
                "schema_version": 1,
                "correction": correction,
                "n_calibration": 9,
                "quantiles": {"c800000": {"coverage": 0.8, "rank": 8, "half_width": 2.0}},
            },
        }
        state["payload_sha256"] = c.digest(state)
        p["postprocessing"] = {"state": state, "calibration_id": "regcal_test", "sha256": "1" * 64}
        p["input_hashes"]["regression_calibration"] = "1" * 64
        point = raw * correction["slope"] + correction["intercept"]
    frame = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "y_true": point + [1.0, 1.0, -1.0],
            "y_pred": point,
            "y_pred_raw": raw,
            "residual": [1.0, 1.0, -1.0],
            "absolute_error": [1.0, 1.0, 1.0],
        }
    )
    if method is not None:
        frame["pi_lower_c800000"] = point - 2
        frame["pi_upper_c800000"] = point + 2
    metrics = [
        {
            "variant": cfg["release_variant"],
            "metric": k,
            "value": v,
            "direction": direction,
            "status": "defined",
            "uncertainty": "not_requested",
        }
        for k, v, direction in [("rmse", 1.0, "min"), ("mae", 1.0, "min"), ("r2", -0.5, "max")]
    ]
    if method is not None:
        metrics += [
            {
                "variant": cfg["release_variant"],
                "metric": k + "_c800000",
                "value": v,
                "direction": "diagnostic",
                "status": "defined",
                "uncertainty": "split_conformal_prediction_interval",
            }
            for k, v in [("empirical_coverage", 1.0), ("mean_width", 4.0)]
        ]
    identity = c.digest({"spec": cfg, "provenance": p})
    summary = {
        "model_type": "nn",
        "nn_evaluation_schema": 1,
        "task": "regression",
        "automatic_promotion": False,
        "n_rows": 3,
        "evaluation_id": "eval_test",
        "parent_run_id": "run_test",
        "identity_hash": identity,
        "predictor_kind": "final",
        "release_variant": cfg["release_variant"],
        "calibration_id": cfg["calibration_id"],
        "target_unit": "MPa",
        "interval_coverages": [0.8] if method is not None else [],
        "criteria_status": "not_assessed",
        "primary_metric": "rmse",
        "primary_metric_direction": "min",
        "primary_value": 1.0,
        "prior_exposure": False,
    }
    data = {
        "summary": summary,
        "spec": cfg,
        "provenance": p,
        "metrics": metrics,
        "acceptance": {"status": "not_assessed", "results": [], "automatic_promotion": False},
        "residual_summary": {"mean": 1 / 3},
        "prior_exposures": [],
    }
    record = {
        "status": "succeeded",
        "accessed_at": "recorded",
        "evaluation_id": "eval_test",
        "parent_run_id": "run_test",
        "identity_hash": identity,
        "silver_id": "silver_test",
        "holdout_fingerprint": fp,
        "calibration_id": cfg["calibration_id"],
        "spec_json": json.dumps(cfg),
        "prior_json": "[]",
    }
    return (data, record, frame)


@pytest.mark.parametrize("method", [None, "none", "bias", "affine"])
def test_frozen_report_and_predictions(method):
    data, record, frame = evidence(method)
    c.validate_report(data, record, copy.deepcopy(data["metrics"]))
    c.validate_predictions(frame, data)


@pytest.mark.parametrize(
    "kind",
    [
        "status",
        "access",
        "identity",
        "unit",
        "metrics",
        "promotion",
        "marker",
        "descriptor",
        "duplicate",
        "prior",
    ],
)
def test_invalid_evidence(kind):
    data, record, _ = evidence()
    stored = copy.deepcopy(data["metrics"])
    if kind == "status":
        record["status"] = "running"
    elif kind == "access":
        record["accessed_at"] = None
    elif kind == "identity":
        record["identity_hash"] = "wrong"
    elif kind == "unit":
        data["summary"]["target_unit"] = "wrong"
    elif kind == "metrics":
        stored[0]["value"] = 99.0
    elif kind == "promotion":
        data["summary"]["automatic_promotion"] = True
    elif kind == "marker":
        data["summary"]["nn_evaluation_schema"] = True
    elif kind == "descriptor":
        data["provenance"]["predictor"]["members"][0]["weight"] = 0.5
    elif kind == "duplicate":
        data["metrics"].append(copy.deepcopy(data["metrics"][0]))
    else:
        record["prior_json"] = '[{"source":"other"}]'
    with pytest.raises(ValueError):
        c.validate_report(data, record, stored)


@pytest.mark.parametrize(
    "kind", ["unit", "rank", "width", "payload", "unselected", "disabled_levels"]
)
def test_bad_interval_state(kind):
    data, _, _ = evidence("bias")
    state = data["provenance"]["postprocessing"]["state"]
    if kind == "unit":
        state["config"]["target_unit"] = "wrong"
    elif kind == "rank":
        state["parameters"]["quantiles"]["c800000"]["rank"] = 7
    elif kind == "width":
        state["parameters"]["quantiles"]["c800000"]["half_width"] = -1
    elif kind == "payload":
        state["payload_sha256"] = "bad"
    elif kind == "unselected":
        data["spec"]["uncertainty"]["coverages"] = [0.9]
    else:
        data["spec"]["uncertainty"]["enabled"] = False
    if kind in ["unit", "rank", "width"]:
        state["payload_sha256"] = c.digest(
            {k: v for k, v in state.items() if k != "payload_sha256"}
        )
    with pytest.raises(ValueError):
        c.interval_state(data["spec"], data["provenance"])


@pytest.mark.parametrize(
    "kind", ["keys", "duplicate", "residual", "point", "bounds", "nan", "extra", "missing_raw"]
)
def test_prediction_integrity(kind):
    data, _, frame = evidence("bias")
    if kind == "keys":
        frame.loc[0, "id"] = 99
    elif kind == "duplicate":
        frame.loc[0, "id"] = 11
    elif kind == "residual":
        frame.loc[0, "residual"] = 99.0
    elif kind == "point":
        frame.loc[0, "y_pred_raw"] = 99.0
    elif kind == "bounds":
        frame.loc[0, "pi_lower_c800000"] -= 1
    elif kind == "nan":
        frame.loc[0, "y_true"] = np.nan
    elif kind == "extra":
        frame["unexpected"] = 1
    else:
        frame = frame.drop(columns="y_pred_raw")
    with pytest.raises(ValueError):
        c.validate_predictions(frame, data)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "x/../y", "x\\y", ""])
def test_paths(tmp_path, name):
    with pytest.raises(ValueError):
        c.safe(tmp_path, name)


def test_no_ddl_in_metadata_reads():
    connection = sqlite3.connect(":memory:")
    assert nn.cached(connection, "x") is None and nn.record(connection, "x") is None
    assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    connection.close()


@pytest.mark.parametrize(
    "case",
    ["not_accessed", "no_lock", "wrong_identity", "wrong_keys", "wrong_purpose", "authorized"],
)
def test_committed_authorization(case, monkeypatch):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE final_evaluations (evaluation_id TEXT,status TEXT,accessed_at TEXT,parent_run_id TEXT,identity_hash TEXT,active_identity TEXT,holdout_fingerprint TEXT,spec_json TEXT)"
    )
    connection.execute("CREATE TABLE final_evaluation_locks (silver_id TEXT,evaluation_id TEXT)")
    payload = {"parent_run_id": "run_test", "expected_silver_dataset_version_id": "silver_test"}
    spec = SimpleNamespace(**payload)
    spec.model_dump = lambda **kwargs: payload.copy()
    connection.execute(
        "INSERT INTO final_evaluations VALUES (?,?,?,?,?,?,?,?)",
        (
            "eval",
            "running",
            None if case == "not_accessed" else "committed",
            "run_test",
            "identity",
            "identity",
            "population",
            json.dumps(payload),
        ),
    )
    if case != "no_lock":
        connection.execute("INSERT INTO final_evaluation_locks VALUES ('silver_test','eval')")
    connection.commit()
    p = SimpleNamespace(
        spec=spec,
        purpose="predictor_only" if case == "wrong_purpose" else "nn_evaluation",
        identity_hash="bad" if case == "wrong_identity" else "identity",
        fingerprint="population",
        key_col="id",
        hold_keys=[10, 11, 99] if case == "wrong_keys" else [10, 11, 12],
        assignments=pd.DataFrame({"id": [10, 11, 12], "partition": ["holdout"] * 3}),
    )
    seen = []
    module = types.ModuleType("ml_template.features.prepared_recipe")

    def load(prepared, keys):
        seen.append(keys)
        return (
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=keys),
            pd.Series([1.0, 2.0, 3.0], index=keys),
        )

    module.load_partition = load
    monkeypatch.setitem(sys.modules, "ml_template.features.prepared_recipe", module)
    if case == "authorized":
        _, y = nn.load_authorized_holdout(connection, p, "eval")
        assert y.tolist() == [1.0, 2.0, 3.0] and seen == [[10, 11, 12]]
    else:
        with pytest.raises(ValueError):
            nn.load_authorized_holdout(connection, p, "eval")
        assert not seen
    connection.close()


def test_calibration_binding_unchanged_by_evaluator_files():
    data, _, _ = evidence()
    p = data["provenance"]
    binding = c.prediction_binding(p)
    p["code_hashes"].update(
        {
            "evaluation/nn_evaluate.py": "new",
            "evaluation/regression_evaluate.py": "new",
            "tracking/nn_evaluation_contract.py": "new",
            "dashboard/regression_multi_data.py": "new",
        }
    )
    assert c.prediction_binding(p) == binding


def write_document(root):
    data, _, _ = evidence()
    for key in ["summary", "spec", "provenance", "acceptance", "residual_summary"]:
        (root / (key + ".json")).write_text(json.dumps(data[key]))
    (root / "report_data.json").write_text(json.dumps(data))
    for name in ["metrics.csv", "report.html", "holdout_predictions.parquet"]:
        (root / name).write_bytes(b"hash-only test fixture; no Parquet decoding")
    entries = {p.name: {"sha256": c.sha256(p), "bytes": p.stat().st_size} for p in root.iterdir()}
    (root / "artifact_manifest.json").write_text(json.dumps({"files": entries}))
    return data


def test_hash_only_metadata_verification(tmp_path):
    data = write_document(tmp_path)
    assert c.read_document(tmp_path) == data


def test_corrupt_frozen_predictions_rejected_before_decode(tmp_path):
    write_document(tmp_path)
    (tmp_path / "holdout_predictions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        c.read_document(tmp_path)
