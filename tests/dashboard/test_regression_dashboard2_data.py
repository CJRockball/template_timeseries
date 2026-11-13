"""Part 2 regression cohorts, ranking and frozen evidence without execution APIs."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import regression_data as base
from ml_template.dashboard import regression_multi_data as data


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
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE model_runs(run_id TEXT,status TEXT,artifact_dir TEXT,config_hash TEXT,silver_dataset_version_id TEXT,split_id TEXT,model_type TEXT,seed INTEGER,feature_set_id TEXT,primary_metric TEXT,primary_value REAL,experiment_name TEXT,experiment_run_id TEXT,git_sha TEXT,git_dirty INTEGER,started_at TEXT,duration_s REAL,error_summary TEXT); CREATE TABLE run_metrics(run_id TEXT,split TEXT,fold INTEGER,metric TEXT,value REAL); CREATE TABLE experiment_launches(experiment_run_id TEXT,experiment_name TEXT,kind TEXT,n_runs INTEGER,status TEXT,started_at TEXT); CREATE TABLE final_evaluations(evaluation_id TEXT,parent_run_id TEXT,status TEXT,identity_hash TEXT,artifact_dir TEXT,spec_json TEXT,silver_id TEXT,holdout_fingerprint TEXT,accessed_at TEXT,calibration_id TEXT); CREATE TABLE final_evaluation_metrics(evaluation_id TEXT,result_json TEXT);"
    )
    conn.execute(
        "INSERT INTO experiment_launches VALUES ('launch','exp','sweep',5,'succeeded','2026-11-05T00:00:00+00:00')"
    )
    tables, roots, reads = {}, {}, []
    analyses = tmp_path / "analyses"
    analyses.mkdir()
    monkeypatch.setattr(base, "analysis_root", lambda: analyses)
    original = pd.read_parquet

    def parquet(path, frame):
        path.write_bytes(("fixture parquet " + path.name).encode())
        tables[path.resolve()] = frame.copy(deep=True)

    def read(path, columns=None, **kwargs):
        path = Path(path).resolve()
        assert path in tables, "Unexpected source dataset/model read"
        reads.append(path)
        frame = tables[path].copy(deep=True)
        return frame if columns is None else frame[columns]

    monkeypatch.setattr(pd, "read_parquet", read)

    def run(
        rid,
        score=1.0,
        seed=1,
        unit="MPa",
        depth=3,
        dirty=0,
        split="split",
        status="succeeded",
        task="regression",
        environment=True,
    ):
        root = tmp_path / rid
        root.mkdir()
        cfg = {
            "schema_version": 2,
            "task": task,
            "seed": seed,
            "experiment_name": "exp",
            "data": {
                "dataset_name": "concrete",
                "silver_dataset_version_id": "silver",
                "feature_set_id": None,
            },
            "split": {"n_folds": 2, "method": "random", "seed": 42, "cv_seed": None},
            "model": {
                "type": "xgb",
                "params": {"objective": "reg:squarederror", "max_depth": depth},
            },
            "metrics": {"primary": "rmse"},
            "fitted_features": None,
            "analysis": {},
        }
        dump(root / "config.json", cfg)
        dump(
            root / "features.json",
            {
                "features": ["x"],
                "silver_dataset_version_id": "silver",
                "split_id": split,
                "dtypes": {"x": "float64"},
            },
        )
        if environment:
            dump(
                root / "environment.json",
                {"python": "3.11", "packages": {"xgboost": "2.0"}, "platform": "fixture"},
            )
        (root / "fold_models").mkdir()
        (root / "model.ubj").write_bytes(b"never deserialize")
        for fold in [0, 1]:
            (root / f"fold_models/fold_{fold}.ubj").write_bytes((rid + str(fold)).encode())
        pd.DataFrame({"fold": [0, 1], "best_round": [2, 2]}).to_csv(
            root / "cv_folds.csv", index=False
        )
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        oof = pd.DataFrame({"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "y_pred_oof": truth + score})
        parquet(root / "oof_predictions.parquet", oof)
        seal(root)
        started = f"2026-11-05T00:{len(roots):02d}:00+00:00"
        metric = "rmse" if task == "regression" else "pr_auc"
        conn.execute(
            "INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid,
                status,
                str(root),
                base.snapshot_hash(cfg),
                "silver",
                split,
                "xgb",
                seed,
                None,
                metric,
                score,
                "exp",
                "launch",
                "a" * 40,
                dirty,
                started,
                5.0,
                None,
            ),
        )
        for name, value in [(metric, score), ("mae", score), ("r2", 1 - score * score / 1.25)]:
            for fold in [-1, 0, 1]:
                conn.execute(
                    "INSERT INTO run_metrics VALUES (?,?,?,?,?)", (rid, "cv", fold, name, value)
                )
        conn.commit()
        if unit is not None and task == "regression":
            directory = analyses / ("analysis_" + rid)
            directory.mkdir()
            r = oof.copy()
            r["y_true"] = truth
            r["residual"] = truth - r.y_pred_oof
            r["absolute_error"] = r.residual.abs()
            r["squared_error"] = r.residual**2
            r["error_kind"] = np.where(
                r.residual > 0,
                "underprediction",
                np.where(r.residual < 0, "overprediction", "exact"),
            )
            parquet(directory / "oof_residuals.parquet", r)
            r.iloc[:2].assign(x=[10.0, 20.0]).to_csv(directory / "error_sample.csv", index=False)
            pd.DataFrame(
                columns=[
                    "feature",
                    "fold",
                    "mean",
                    "repeat_std",
                    "between_fold_std",
                    "metric",
                    "target_unit",
                ]
            ).to_csv(directory / "permutation_importance.csv", index=False)
            summary = {
                "n_development": 4,
                "target_unit": unit,
                "holdout_labels_loaded": False,
                "permutation": {"enabled": False, "metric": "rmse"},
            }
            dump(directory / "report_data.json", {"summary": summary})
            dump(directory / "residual_summary.json", summary)
            dump(
                directory / "analysis.json",
                {"analysis_id": directory.name, "parent_run_id": rid, "status": "succeeded"},
            )
            entry = base.manifest(root)
            names = ["cv_folds.csv", "fold_models/fold_0.ubj", "fold_models/fold_1.ubj"]
            desc = {
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
                "files_sha256": {n: entry[n]["sha256"] for n in names},
            }
            dump(
                directory / "provenance.json",
                {
                    "parent_run_id": rid,
                    "fold_descriptor": desc,
                    "input_provenance": {
                        "input_hashes": {
                            "config": entry["config.json"]["sha256"],
                            "features": entry["features.json"]["sha256"],
                        }
                    },
                },
            )
            seal(directory)
        roots[rid] = root
        return root

    def evaluation(eid="eval_one", parent="a"):
        directory = tmp_path / eid
        directory.mkdir()
        spec = {
            "schema_version": 2,
            "task": "regression",
            "parent_run_id": parent,
            "predictor": {"kind": "final", "weighting": "equal"},
            "primary_metric": "rmse",
            "expected_silver_dataset_version_id": "silver",
            "expected_split_id": "split",
            "calibration_id": None,
            "release_variant": "raw",
            "uncertainty": {"enabled": False},
            "criteria": [],
        }
        fingerprint = data.population_fingerprint("silver", "id", [11, 12, 13, 14])
        provenance = {
            "holdout_fingerprint": fingerprint,
            "predictor": {
                "schema_version": 2,
                "task": "regression",
                "prediction_kind": "continuous",
                "kind": "final",
            },
            "input_hashes": {},
            "code_hashes": {},
        }
        identity = data.digest({"spec": spec, "provenance": provenance})
        summary = {
            "evaluation_id": eid,
            "identity_hash": identity,
            "task": "regression",
            "parent_run_id": parent,
            "predictor_kind": "final",
            "n_rows": 4,
            "primary_metric": "rmse",
            "primary_metric_direction": "min",
            "primary_value": 0.5,
            "criteria_status": "not_assessed",
            "prior_exposure": False,
            "automatic_promotion": False,
        }
        metrics = [
            {
                "variant": "raw",
                "metric": name,
                "value": value,
                "direction": data.direction(name),
                "status": "defined",
                "uncertainty": "not_requested",
            }
            for name, value in [("rmse", 0.5), ("mae", 0.5), ("r2", 0.8)]
        ]
        acceptance = {"status": "not_assessed", "results": [], "automatic_promotion": False}
        residual_summary = {
            "convention": "y_true - y_pred",
            "mean": -0.5,
            "std_population": 0.0,
            "quantiles": {"0.5": -0.5},
        }
        report = {
            "summary": summary,
            "spec": spec,
            "provenance": provenance,
            "metrics": metrics,
            "acceptance": acceptance,
            "residual_summary": residual_summary,
            "warnings": [],
        }
        for name, value in [
            ("report_data", report),
            ("summary", summary),
            ("spec", spec),
            ("provenance", provenance),
            ("acceptance", acceptance),
            ("residual_summary", residual_summary),
        ]:
            dump(directory / (name + ".json"), value)
        pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
        parquet(
            directory / "holdout_predictions.parquet",
            pd.DataFrame(
                {
                    "id": [11, 12, 13, 14],
                    "y_true": [1.0, 2.0, 3.0, 4.0],
                    "y_pred": [1.5, 2.5, 3.5, 4.5],
                    "residual": [-0.5] * 4,
                    "absolute_error": [0.5] * 4,
                }
            ),
        )
        seal(directory)
        conn.execute(
            "INSERT INTO final_evaluations VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                eid,
                parent,
                "succeeded",
                identity,
                str(directory),
                json.dumps(spec),
                "silver",
                fingerprint,
                "2026-11-05T07:18:29+00:00",
                None,
            ),
        )
        for row in metrics:
            conn.execute(
                "INSERT INTO final_evaluation_metrics VALUES (?,?)", (eid, json.dumps(row))
            )
        conn.commit()
        return directory

    yield {
        "engine": conn,
        "run": run,
        "evaluation": evaluation,
        "roots": roots,
        "tables": tables,
        "reads": reads,
        "analyses": analyses,
        "original_reader": original,
    }
    conn.close()


def test_metric_directions_and_ranking(world):
    world["run"]("a", score=2.0, depth=3)
    world["run"]("b", score=1.0, depth=4)
    records, _ = data.inventory(world["engine"])
    board = data.rank_recipes(records)
    assert board.iloc[0]["mean"] == 1.0 and board.direction.eq("min").all()
    changed = copy.copy(records)
    changed = [
        {**r, "primary_metric": "r2", "primary_value": i * 0.2} for i, r in enumerate(changed)
    ]
    assert data.rank_recipes(changed).iloc[0]["mean"] == 0.2


def test_latest_seed_repeat_not_counted_twice(world):
    world["run"]("a", score=2.0, seed=1)
    world["run"]("b", score=4.0, seed=2)
    world["run"]("c", score=1.0, seed=1)
    rows, _ = data.inventory(world["engine"])
    board = data.rank_recipes(rows)
    assert len(board) == 1 and board.iloc[0].n_seeds == 2
    assert board.iloc[0]["mean"] == 2.5 and board.iloc[0].excluded_repeats == 1


def test_dirty_or_missing_environment_not_pooled(world):
    world["run"]("a", seed=1, dirty=1)
    world["run"]("b", seed=2, dirty=1)
    world["run"]("c", seed=3, environment=False)
    records, _ = data.inventory(world["engine"])
    board = data.rank_recipes(records)
    assert len(board) == 3 and board.n_seeds.eq(1).all()


def test_units_inherit_only_with_unambiguous_same_silver(world):
    world["run"]("a", unit="MPa")
    world["run"]("b", unit=None)
    records, _ = data.inventory(world["engine"])
    assert {r["target_unit"] for r in records} == {"MPa"}
    world["run"]("c", unit="Pa")
    records, _ = data.inventory(world["engine"])
    assert not data.cohort_options(records)
    with pytest.raises(ValueError):
        data.comparison(world["engine"], ["a", "b"])


def test_split_and_task_mismatch_blocked(world):
    world["run"]("a")
    world["run"]("b", split="another")
    world["run"]("binary", task="binary_classification")
    with pytest.raises(ValueError):
        data.comparison(world["engine"], ["a", "b"])
    with pytest.raises(ValueError):
        data.comparison(world["engine"], ["a", "binary"])


def test_comparison_deltas_and_explicit_analyses(world):
    world["run"]("a", score=2.0)
    world["run"]("b", score=1.0)
    result = data.comparison(world["engine"], ["a", "b"], metric="rmse", baseline="a")
    assert result["point"].set_index("run_id").loc["b", "improvement_vs_baseline"] == 1.0
    assert not result["residuals"]
    result = data.comparison(
        world["engine"],
        ["a", "b"],
        analyses={"a": "analysis:analysis_a", "b": "analysis:analysis_b"},
    )
    assert result["all_analyses_selected"]


def test_key_mismatch_no_intersection_fallback(world):
    world["run"]("a")
    root = world["run"]("b")
    world["tables"][(root / "oof_predictions.parquet").resolve()].loc[0, "id"] = 999
    with pytest.raises(ValueError, match="coverage differs"):
        data.comparison(world["engine"], ["a", "b"])


def test_missing_metric_not_silently_dropped(world):
    world["run"]("a")
    world["run"]("b")
    world["engine"].execute("DELETE FROM run_metrics WHERE run_id='b' AND metric='mae'")
    result = data.comparison(world["engine"], ["a", "b"], metric="mae")
    assert len(result["point"]) == 2 and pd.isna(
        result["point"].set_index("run_id").loc["b", "value"]
    )


def test_frozen_metadata_does_not_read_predictions_or_write(world):
    world["evaluation"]()
    engine = world["engine"]
    before = engine.total_changes
    denied = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
    }
    engine.set_authorizer(
        lambda action, *args: sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK
    )
    loaded = data.load_regression_evaluation(engine, "eval_one")
    assert loaded["state"] == "verified", loaded.get("message")
    assert loaded["predictions"] is None and not world["reads"]
    assert engine.total_changes == before


def test_saved_predictions_explicit_and_population_checked(world):
    root = world["evaluation"]()
    result = data.load_regression_evaluation(world["engine"], "eval_one", predictions=True)
    assert result["state"] == "verified" and len(result["predictions"]) == 4
    world["tables"][(root / "holdout_predictions.parquet").resolve()].loc[0, "id"] = 999
    assert (
        data.load_regression_evaluation(world["engine"], "eval_one", predictions=True)["state"]
        == "invalid"
    )


def test_tampered_frozen_metrics_rejected(world):
    root = world["evaluation"]()
    report = base.read_json(root / "report_data.json")
    report["metrics"][0]["value"] = 0.01
    dump(root / "report_data.json", report)
    seal(root)
    assert data.load_regression_evaluation(world["engine"], "eval_one")["state"] == "invalid"


def test_sweep_progress_global_counts_not_mislabelled(world):
    world["run"]("a")
    world["run"]("b", status="failed")
    world["run"]("binary", task="binary_classification")
    result = data.sweep_progress(world["engine"], "launch")
    assert result["registered_runs_all_tasks"] == 3 and result["regression_registered"] == 2
    assert result["failed"] == 1 and result["unregistered_all_tasks"] == 2


def test_real_parquet_when_available(world, monkeypatch):
    pytest.importorskip("pyarrow")
    world["run"]("a", score=2.0)
    world["run"]("b", score=1.0)
    for path, frame in world["tables"].items():
        frame.to_parquet(path, index=False)
    for root in world["roots"].values():
        seal(root)
    for root in world["analyses"].iterdir():
        seal(root)
    monkeypatch.setattr(pd, "read_parquet", world["original_reader"])
    assert data.comparison(world["engine"], ["a", "b"])["point"].shape[0] == 2


def test_binary_views_filter_regression_and_mixed_expected_counts(world, monkeypatch):
    from ml_template.dashboard import experiment_data, sweep_data

    world["run"]("a")
    world["run"]("binary", task="binary_classification")
    frame = pd.DataFrame(data.all_records(world["engine"]))
    monkeypatch.setattr(experiment_data, "load_experiment_runs", lambda engine: frame.copy())
    assert data.binary_runs(world["engine"]).run_id.tolist() == ["binary"]
    launch = pd.DataFrame(data.launches(world["engine"]))
    params = pd.DataFrame({"run_id": ["a", "binary"], "key": ["seed", "seed"], "value": ["1", "1"]})
    monkeypatch.setattr(
        sweep_data, "load_sweep_data", lambda engine: (launch.copy(), frame.copy(), params.copy())
    )
    launched, runs, recorded = data.binary_sweep_data(world["engine"])
    assert runs.run_id.tolist() == ["binary"] and recorded.run_id.tolist() == ["binary"]
    assert pd.isna(launched.n_runs.iloc[0])
    assert data.nonbinary_launch(world["engine"], "launch")
