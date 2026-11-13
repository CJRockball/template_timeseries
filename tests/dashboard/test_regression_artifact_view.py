"""Saved regression dashboard tests; no fitting or source dataset reads."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import regression_data as data
from ml_template.dashboard import regression_figs as figures


def dump(path, value):
    path.write_text(json.dumps(value))


def seal(root):
    dump(
        root / "artifact_manifest.json",
        {
            "files": {
                p.relative_to(root).as_posix(): {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "bytes": p.stat().st_size,
                }
                for p in root.rglob("*")
                if p.is_file() and p.name != "artifact_manifest.json"
            }
        },
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    cfg = {
        "schema_version": 2,
        "task": "regression",
        "seed": 42,
        "experiment_name": "test",
        "data": {"silver_dataset_version_id": "silver", "feature_set_id": None},
        "model": {"type": "xgb"},
        "split": {"n_folds": 2, "method": "random", "cv_seed": None},
        "fitted_features": None,
        "analysis": {},
    }
    dump(root / "config.json", cfg)
    dump(
        root / "features.json",
        {"features": ["x"], "silver_dataset_version_id": "silver", "split_id": "split"},
    )
    (root / "fold_models").mkdir()
    for k in [0, 1]:
        (root / f"fold_models/fold_{k}.ubj").write_bytes(b"hash bytes only; never deserialize")
    pd.DataFrame({"fold": [0, 1], "best_round": [2, 2]}).to_csv(root / "cv_folds.csv", index=False)
    tables = {}

    def parquet(path, frame):
        path.write_bytes(("fixture parquet: " + path.name).encode())
        tables[path.resolve()] = frame.copy(deep=True)

    oof = pd.DataFrame(
        {"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "y_pred_oof": [1.0, 2.0, 3.0, 4.0]}
    )
    parquet(root / "oof_predictions.parquet", oof)
    records = []
    for k in [0, 1]:
        for dataset in ["inner_fit", "inner_stop"]:
            for rnd, value in enumerate([3.0, 2.0, 2.1], 1):
                records.append(
                    {
                        "fold": k,
                        "round": rnd,
                        "dataset": dataset,
                        "metric": "rmse",
                        "value": value,
                        "selected_round": 2,
                        "direction": "min",
                        "curve_role": "inner_early_stopping",
                    }
                )
    inner = pd.DataFrame(records)
    outer = inner.rename(columns={"selected_round": "best_round"}).replace(
        {"dataset": {"inner_fit": "train", "inner_stop": "val"}}
    )
    outer = outer[["fold", "round", "dataset", "metric", "value", "best_round"]]
    parquet(root / "early_stopping_curves.parquet", inner)
    parquet(root / "learning_curves.parquet", outer)
    dump(
        root / "early_stopping_curves_metadata.json",
        {"curve_role": "inner_early_stopping", "used_holdout": False},
    )
    dump(
        root / "learning_curves_metadata.json",
        {"curve_role": "outer_diagnostic_refit", "used_holdout": False},
    )
    seal(root)
    entries = data.manifest(root)
    analysis_root = tmp_path / "analyses"
    analysis_root.mkdir()
    analysis = analysis_root / "analysis_good"
    analysis.mkdir()
    residual = oof.copy()
    residual["y_true"] = [1.5, 1.0, 3.0, 5.0]
    residual["residual"] = residual.y_true - residual.y_pred_oof
    residual["absolute_error"] = residual.residual.abs()
    residual["squared_error"] = residual.residual**2
    residual["error_kind"] = ["underprediction", "overprediction", "exact", "underprediction"]
    parquet(analysis / "oof_residuals.parquet", residual)
    residual.iloc[[1, 3]].assign(x=[10.0, 20.0]).to_csv(analysis / "error_sample.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["x"] * 3,
            "fold": [0, 1, -1],
            "mean": [2.0, 4.0, 3.0],
            "repeat_std": [0.5, 0.7, np.nan],
            "between_fold_std": [np.nan, np.nan, np.sqrt(2)],
            "metric": ["rmse"] * 3,
            "target_unit": ["MPa"] * 3,
        }
    ).to_csv(analysis / "permutation_importance.csv", index=False)
    summary = {
        "n_development": 4,
        "target_unit": "MPa",
        "holdout_labels_loaded": False,
        "permutation": {"enabled": True, "metric": "rmse"},
    }
    dump(analysis / "report_data.json", {"summary": summary})
    dump(analysis / "residual_summary.json", summary)
    dump(
        analysis / "analysis.json",
        {"analysis_id": "analysis_good", "parent_run_id": "reg", "status": "succeeded"},
    )
    names = ["cv_folds.csv", "fold_models/fold_0.ubj", "fold_models/fold_1.ubj"]
    descriptor = {
        "schema_version": 2,
        "kind": "cv_mean",
        "task": "regression",
        "prediction_kind": "continuous",
        "fitted_features": False,
        "weighting": "equal",
        "members": [
            {
                "fold_id": k,
                "model_path": f"fold_models/fold_{k}.ubj",
                "transformer_path": None,
                "weight": 0.5,
            }
            for k in [0, 1]
        ],
        "files_sha256": {n: entries[n]["sha256"] for n in names},
    }
    dump(
        analysis / "provenance.json",
        {
            "parent_run_id": "reg",
            "fold_descriptor": descriptor,
            "input_provenance": {
                "input_hashes": {
                    "config": entries["config.json"]["sha256"],
                    "features": entries["features.json"]["sha256"],
                }
            },
        },
    )
    seal(analysis)
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE model_runs(run_id TEXT,status TEXT,artifact_dir TEXT,config_hash TEXT,silver_dataset_version_id TEXT,split_id TEXT,model_type TEXT,seed INTEGER,feature_set_id TEXT,primary_metric TEXT);CREATE TABLE run_metrics(run_id TEXT,split TEXT,fold INTEGER,metric TEXT,value REAL);"
    )
    conn.execute(
        "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "reg",
            "succeeded",
            str(root),
            data.snapshot_hash(cfg),
            "silver",
            "split",
            "xgb",
            42,
            None,
            "rmse",
        ),
    )
    binary = tmp_path / "binary"
    binary.mkdir()
    dump(binary / "config.json", {"task": "binary_classification"})
    conn.execute(
        "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "binary",
            "succeeded",
            str(binary),
            "legacy",
            "silver_b",
            "split_b",
            "xgb",
            42,
            None,
            "pr_auc",
        ),
    )
    conn.executemany(
        "INSERT INTO run_metrics VALUES (?,?,?,?,?)",
        [
            ("reg", "cv", -1, "rmse", 0.75),
            ("reg", "cv", 0, "rmse", 0.79),
            ("reg", "cv", 1, "rmse", 0.71),
        ],
    )
    conn.commit()
    original = pd.read_parquet
    reads = []

    def read(path, columns=None, **kwargs):
        path = Path(path).resolve()
        assert path in tables, "Attempted source dataset or unapproved Parquet read"
        reads.append(path)
        frame = tables[path].copy(deep=True)
        return frame if columns is None else frame[columns]

    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(data, "analysis_root", lambda: analysis_root)
    yield {
        "engine": conn,
        "root": root,
        "analysis": analysis,
        "tables": tables,
        "reads": reads,
        "inner": inner,
        "outer": outer,
        "original_reader": original,
    }
    conn.close()


def test_explicit_selection_no_default_residual_read(world):
    result = data.load_bundle(world["engine"], "reg")
    assert result["state"] == "verified" and result["analysis"] is None
    assert [x["value"] for x in result["options"]] == ["analysis:analysis_good"]
    assert not any(p.name == "oof_residuals.parquet" for p in world["reads"])


def test_selected_analysis_bound_to_original_oof(world):
    result = data.load_bundle(world["engine"], "reg", "analysis:analysis_good")
    assert result["state"] == "verified", result.get("message")
    assert result["analysis"]["summary"]["target_unit"] == "MPa"
    assert result["analysis"]["residuals"].id.tolist() == [1, 2, 3, 4]


def test_database_and_artifacts_not_modified(world):
    conn = world["engine"]
    before = conn.total_changes
    denied = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
    }
    conn.set_authorizer(
        lambda action, *args: sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK
    )
    hashes = {str(p): data.sha256(p) for p in world["root"].rglob("*") if p.is_file()}
    assert data.load_bundle(conn, "reg", "analysis:analysis_good")["state"] == "verified"
    assert conn.total_changes == before
    assert hashes == {str(p): data.sha256(p) for p in world["root"].rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "name", ["oof_residuals.parquet", "permutation_importance.csv", "report_data.json"]
)
def test_tampered_analysis_not_displayed(world, name):
    (world["analysis"] / name).write_bytes(b"tampered")
    assert data.load_bundle(world["engine"], "reg", "analysis:analysis_good")["state"] == "invalid"


def test_parent_and_path_selection_guards(world):
    assert data.load_bundle(world["engine"], "reg", "analysis:../outside")["state"] == "invalid"
    assert (
        data.load_bundle(world["engine"], "binary", "analysis:analysis_good")["state"] == "invalid"
    )


def test_wrong_parent_even_if_manifest_rebuilt(world):
    path = world["analysis"] / "provenance.json"
    value = data.read_json(path)
    value["parent_run_id"] = "other"
    dump(path, value)
    seal(world["analysis"])
    assert data.load_bundle(world["engine"], "reg", "analysis:analysis_good")["state"] == "invalid"


def test_guard_before_regression_source_labels(world):
    data.guard_binary_labels(data.header(world["engine"], "binary"))
    with pytest.raises(ValueError, match="blocked"):
        data.guard_binary_labels(data.header(world["engine"], "reg"))
    assert not world["reads"]


def test_conflicting_task_does_not_fall_back_to_binary(world):
    path = world["root"] / "config.json"
    cfg = data.read_json(path)
    cfg["task"] = "binary_classification"
    dump(path, cfg)
    assert data.is_regression_run(world["engine"], "reg")
    with pytest.raises(ValueError):
        data.guard_binary_labels(data.header(world["engine"], "reg"))
    assert data.load_bundle(world["engine"], "reg")["state"] == "invalid"
    assert not world["reads"]


def test_markers_only_on_inner_curves(world):
    inner = figures.curves(world["inner"], "inner", "MPa")
    outer = figures.curves(world["outer"], "outer", "MPa")
    assert len(inner.layout.shapes) == 2 and (not outer.layout.shapes)
    assert {t.name for t in outer.data} == {"outer_train", "outer_validation"}


def test_bad_curve_role_rejected(world):
    broken = world["inner"].copy()
    broken["curve_role"] = "outer_diagnostic_refit"
    with pytest.raises(ValueError):
        data.validate_curves(broken, "inner", [0, 1])


def test_metric_direction_and_negative_importance():
    m = figures.metrics_table(
        pd.DataFrame({"metric": ["rmse", "r2"], "fold": [-1, -1], "value": [4.0, 0.9]}), "MPa"
    ).set_index("metric")
    assert m.loc["rmse", "direction"] == "min" and m.loc["r2", "direction"] == "max"
    imp = pd.DataFrame(
        {
            "feature": ["x"],
            "fold": [-1],
            "mean": [-1.0],
            "between_fold_std": [0.2],
            "repeat_std": [np.nan],
            "metric": ["rmse"],
            "target_unit": ["MPa"],
        }
    )
    assert figures.importance(imp).data[0].x[0] == -1.0


def test_missing_optional_placeholders():
    assert figures.curves(None, "inner", "MPa").layout.annotations
    assert figures.residual_figures(None, "MPa")[0].layout.annotations


def test_inconsistent_residual_arithmetic_rejected(world):
    world["tables"][(world["analysis"] / "oof_residuals.parquet").resolve()].loc[0, "residual"] = (
        999.0
    )
    assert data.load_bundle(world["engine"], "reg", "analysis:analysis_good")["state"] == "invalid"


def test_analysis_settings_do_not_change_snapshot_hash(world):
    cfg = data.read_json(world["root"] / "config.json")
    expected = data.snapshot_hash(cfg)
    cfg["analysis"] = {"different": "report settings"}
    assert data.snapshot_hash(cfg) == expected


def test_real_parquet_when_installed(world, monkeypatch):
    pytest.importorskip("pyarrow")
    for path, frame in world["tables"].items():
        frame.to_parquet(path, index=False)
    seal(world["root"])
    seal(world["analysis"])
    monkeypatch.setattr(pd, "read_parquet", world["original_reader"])
    result = data.load_bundle(world["engine"], "reg", "analysis:analysis_good")
    assert result["state"] == "verified", result.get("message")
