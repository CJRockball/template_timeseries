# threshold-diagnostics-v1
import pandas as pd
import pytest
from sqlalchemy import create_engine

pytest.importorskip("pyarrow")

from ml_template.dashboard import run_data  # noqa: E402
from ml_template.db.tracking_tables import (  # noqa: E402
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_feature_importance,
    run_metrics,
    run_params,
)


def _run(run_id, started, name="exp", status="succeeded"):
    return {
        "run_id": run_id,
        "experiment_run_id": "launch1",
        "experiment_name": name,
        "config_hash": "h",
        "status": status,
        "started_at": started,
        "silver_dataset_version_id": "s",
        "split_id": "sp",
        "model_type": "xgb",
        "seed": 1,
        "primary_metric": "pr_auc",
        "primary_value": 0.8,
    }


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    ensure_tracking_tables(eng)
    with eng.begin() as conn:
        conn.execute(
            experiment_launches.insert(),
            [
                {
                    "experiment_run_id": "launch1",
                    "experiment_name": "exp",
                    "kind": "single",
                    "n_runs": 3,
                    "status": "succeeded",
                    "started_at": "2026-10-15T00:00:00Z",
                }
            ],
        )
        conn.execute(
            model_runs.insert(),
            [
                _run("r1", "2026-10-15T01:00:00Z"),
                _run("r2", "2026-10-15T03:00:00Z"),
                _run("r3", "2026-10-15T05:00:00Z", status="failed"),
            ],
        )
        conn.execute(
            run_params.insert(), [{"run_id": "r2", "key": "model.max_depth", "value": "3"}]
        )
        conn.execute(
            run_metrics.insert(),
            [
                {"run_id": "r2", "split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.8},
                {"run_id": "r2", "split": "cv", "fold": 0, "metric": "pr_auc", "value": 0.7},
            ],
        )
        conn.execute(
            run_feature_importance.insert(),
            [
                {
                    "run_id": "r2",
                    "split": "cv",
                    "fold": -1,
                    "method": "perm",
                    "feature": "a",
                    "value": 0.1,
                    "std": 0.01,
                },
                {
                    "run_id": "r2",
                    "split": "cv",
                    "fold": -1,
                    "method": "perm",
                    "feature": "b",
                    "value": 0.3,
                    "std": 0.02,
                },
            ],
        )
    return eng


def test_latest_run_skips_failed_and_orders_by_start(engine):
    assert run_data.latest_run_id(engine) == "r2"
    assert list(run_data.list_runs(engine)["run_id"]) == ["r2", "r1"]


def test_unknown_run_raises(engine):
    with pytest.raises(KeyError):
        run_data.run_header(engine, "nope")


def test_params_metrics_importance(engine):
    assert run_data.load_params(engine, "r2") == {"model.max_depth": "3"}
    assert set(run_data.load_metrics(engine, "r2")["fold"]) == {-1, 0}
    assert list(run_data.load_importance(engine, "r2")["feature"]) == ["b", "a"]


def test_missing_artifact_returns_none(tmp_path):
    assert run_data.load_learning_curves(tmp_path) is None
    assert run_data.load_config(tmp_path) is None


def test_reads_learning_curve_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "fold": [0],
            "round": [1],
            "dataset": ["train"],
            "metric": ["logloss"],
            "value": [0.5],
            "best_round": [10],
        }
    )
    df.to_parquet(tmp_path / "learning_curves.parquet", index=False)
    assert len(run_data.load_learning_curves(tmp_path)) == 1


def test_run_dir_falls_back_to_runs_folder(tmp_path):
    header = {"run_id": "r9", "artifact_dir": "/does/not/exist"}
    assert run_data.run_dir(header, base=tmp_path) == tmp_path / "r9"


def test_labels_and_oof_join(engine, tmp_path):
    from ml_template.db.tracking_tables import splits

    silver = tmp_path / "silver" / "ai4i" / "s"
    silver.mkdir(parents=True)
    pd.DataFrame({"udi": [1, 2, 3, 4], "machine_failure": [0, 1, 0, 1]}).to_parquet(
        silver / "data.parquet", index=False
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    pd.DataFrame({"udi": [1, 2, 3, 4], "partition": ["dev"] * 4, "fold": [0, 0, 1, 1]}).to_parquet(
        split_dir / "assignments.parquet", index=False
    )
    with engine.begin() as conn:
        conn.execute(
            splits.insert(),
            [
                {
                    "split_id": "sp",
                    "silver_dataset_version_id": "s",
                    "path": str(split_dir),
                    "policy_json": "{}",
                    "key_col": "udi",
                    "target_col": "machine_failure",
                }
            ],
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {"udi": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "p_oof": [0.1, 0.9, 0.2, 0.7]}
    ).to_parquet(run_dir / "oof_predictions.parquet", index=False)
    header = run_data.run_header(engine, "r2")
    labels = run_data.load_labels(engine, header, silver_root=tmp_path / "silver")
    assert labels.loc[2] == 1 and labels.index.name == "udi"
    joined = run_data.load_oof_with_labels(engine, header, run_dir, tmp_path / "silver")
    assert list(joined["y_true"]) == [0, 1, 0, 1]


def test_labels_missing_without_split_row(engine, tmp_path):
    header = run_data.run_header(engine, "r2")
    assert run_data.load_labels(engine, header, silver_root=tmp_path) is None
