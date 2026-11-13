"""Synthetic real-Parquet integration. No model checkpoint exists or is loaded."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stored(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from ml_template.dashboard import regression_data as base
    from ml_template.evaluation.residual_analysis_run import ResidualConfig, split_identity
    from ml_template.tracking.run_artifacts import write_artifact_manifest

    run = tmp_path / "runs/run_test"
    run.mkdir(parents=True)
    silver = tmp_path / "silver/test/silver_test"
    silver.mkdir(parents=True)
    splits = tmp_path / "splits"
    out = tmp_path / "analyses"
    monkeypatch.setenv("ML_TEMPLATE_ANALYSIS_ROOT", str(out))
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: y}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [x], target: [y]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, x: x, y: y}\n"
        "  canonical_dtypes: {id: int64, x: float64, y: float64}\n"
    )
    policy = {
        "method": "random",
        "n_folds": 2,
        "holdout_fraction": 0.2,
        "seed": 42,
        "calibration_fraction": 0.1,
        "recalibration_fraction": 0.1,
    }
    split_id = split_identity("silver_test", policy, "id", "y")
    cfg = {
        "schema_version": 2,
        "task": "regression",
        "experiment_name": "synthetic_nn",
        "seed": 42,
        "data": {
            "dataset_name": "test",
            "dataset_config": str(contract),
            "silver_dataset_version_id": "silver_test",
            "feature_set_id": None,
        },
        "split": {**policy, "cv_seed": None},
        "analysis": {},
        "fitted_features": None,
        "model": {"type": "nn", "params": {"task": "regression"}},
        "metrics": {"primary": "rmse", "names": ["rmse", "mae", "r2"], "alert_rates": []},
    }
    assignments = pd.DataFrame(
        {
            "id": range(10),
            "partition": ["dev"] * 6 + ["holdout"] * 2 + ["calibration", "recalibration"],
            "fold": [0, 1, 0, 1, 0, 1, -1, -1, -1, -1],
        }
    )
    split_dir = splits / split_id
    split_dir.mkdir(parents=True)
    assignments.to_parquet(split_dir / "assignments.parquet", index=False)
    split_manifest = {
        "split_id": split_id,
        "silver_dataset_version_id": "silver_test",
        "policy": policy,
        "key_col": "id",
        "target_col": "y",
        "n_rows": 10,
        "n_holdout": 2,
        "n_calibration": 1,
        "n_recalibration": 1,
        "fold_sizes": {"0": 3, "1": 3},
        "assignments_sha256": hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest(),
    }
    (split_dir / "manifest.json").write_text(json.dumps(split_manifest))
    data = pd.DataFrame(
        {
            "id": range(10),
            "x": np.arange(10, dtype=float),
            "y": [float(k) + 0.5 for k in range(6)] + [np.inf, np.nan, np.inf, np.nan],
        }
    )
    data.to_parquet(silver / "data.parquet", index=False)
    sm = {
        "validation_passed": True,
        "silver_dataset_version_id": "silver_test",
        "roles": {"id": ["id"], "feature": ["x"], "target": ["y"]},
        "config_sha256": base.sha256(contract),
        "parquet_sha256": base.sha256(silver / "data.parquet"),
    }
    (silver / "manifest.json").write_text(json.dumps(sm))
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "features.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "split_id": split_id,
                "features": ["x"],
                "dtypes": {"x": "float64"},
            }
        )
    )
    pd.DataFrame(
        {
            "id": range(6),
            "fold": [0, 1, 0, 1, 0, 1],
            "y_pred_oof": [float(k) + 0.25 for k in range(6)],
        }
    ).to_parquet(run / "oof_predictions.parquet", index=False)
    write_artifact_manifest(run)
    db = tmp_path / "project.db"
    c = sqlite3.connect(db)
    c.execute(
        "CREATE TABLE model_runs (run_id TEXT, status TEXT, artifact_dir TEXT, config_hash TEXT, silver_dataset_version_id TEXT, feature_set_id TEXT, split_id TEXT, seed INTEGER, model_type TEXT)"
    )
    c.execute(
        "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "run_test",
            "succeeded",
            str(run),
            base.snapshot_hash(cfg),
            "silver_test",
            None,
            split_id,
            42,
            "nn",
        ),
    )
    c.commit()
    c.close()
    request = ResidualConfig(
        enabled=True,
        run_id="run_test",
        target_unit="synthetic_units",
        database=str(db),
        silver_root=str(tmp_path / "silver"),
        split_root=str(splits),
        output_root=str(out),
        feature_columns=["x"],
    )
    return {"cfg": request, "run": run, "out": out, "silver": silver, "db": db, "base": base}


def connect(stored):
    return sqlite3.connect(stored["db"].as_uri() + "?mode=ro", uri=True)


def test_real_saved_nn_residual_analysis_and_dashboard_discovery(stored, monkeypatch):
    from ml_template.evaluation.residual_analysis_run import run_residual_analysis

    old = pd.read_parquet
    target_calls = []

    def spy(path, *args, **kwargs):
        if Path(path) == stored["silver"] / "data.parquet":
            assert kwargs["columns"] == ["id", "y", "x"]
            assert set(kwargs["filters"][0][2]) == set(range(6))
            target_calls.append(path)
        return old(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    before = {str(p): stored["base"].sha256(p) for p in stored["run"].rglob("*") if p.is_file()}
    db_hash = stored["base"].sha256(stored["db"])
    conn = connect(stored)
    try:
        checked = run_residual_analysis(conn, stored["cfg"], check_only=True)
        assert checked["ready"] and not target_calls and not stored["out"].exists()
        result = run_residual_analysis(conn, stored["cfg"], check_only=False)
        assert result["status"] == "succeeded" and len(target_calls) == 1
        loaded = stored["base"].load_bundle(conn, "run_test")
        assert loaded["state"] == "verified", loaded
        assert len(loaded["options"]) == 1
        chosen = loaded["options"][0]["value"]
        loaded = stored["base"].load_bundle(conn, "run_test", chosen)
        assert loaded["state"] == "verified", loaded
        analysis = loaded["analysis"]
        assert analysis["importance"].empty
        assert analysis["summary"]["pooled"]["rmse"] == pytest.approx(0.25)
        assert len(analysis["summary"]["fold_summaries"]) == 2
        assert np.allclose(analysis["residuals"].residual, 0.25)
        assert len(target_calls) == 1  # dashboard does not reread source labels
        assert (Path(result["artifact_dir"]) / "fold_summary.csv").is_file()
    finally:
        conn.close()
    assert before == {
        str(p): stored["base"].sha256(p) for p in stored["run"].rglob("*") if p.is_file()
    }
    assert db_hash == stored["base"].sha256(stored["db"])
    assert not (stored["run"] / "model.ubj").exists()


def test_tampered_oof_rejected_before_decode(stored, monkeypatch):
    from ml_template.evaluation.residual_analysis_run import run_residual_analysis

    path = stored["run"] / "oof_predictions.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")
    old = pd.read_parquet

    def reject_decode(candidate, *args, **kwargs):
        if Path(candidate) == path:
            raise AssertionError("corrupt OOF was decoded")
        return old(candidate, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", reject_decode)
    conn = connect(stored)
    try:
        with pytest.raises(ValueError, match="changed"):
            run_residual_analysis(conn, stored["cfg"], check_only=True)
    finally:
        conn.close()


def test_disabled_request_has_no_output(stored):
    from ml_template.evaluation.residual_analysis_run import run_residual_analysis

    cfg = stored["cfg"].model_copy(update={"enabled": False})
    conn = connect(stored)
    try:
        assert not run_residual_analysis(conn, cfg, check_only=True)["ready"]
        with pytest.raises(ValueError, match="disabled"):
            run_residual_analysis(conn, cfg, check_only=False)
    finally:
        conn.close()
    assert not stored["out"].exists()


def test_sealed_run_destination_rejected(stored):
    from ml_template.evaluation.residual_analysis_run import run_residual_analysis

    cfg = stored["cfg"].model_copy(update={"output_root": str(stored["run"])})
    conn = connect(stored)
    try:
        with pytest.raises(ValueError, match="outside"):
            run_residual_analysis(conn, cfg, check_only=False)
    finally:
        conn.close()
