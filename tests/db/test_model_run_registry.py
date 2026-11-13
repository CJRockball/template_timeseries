import math

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from ml_template.db import model_run_registry as reg  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    return sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def _start(engine, launch, run_id="run_1", name="exp"):
    reg.start_model_run(
        engine,
        run_id=run_id,
        experiment_run_id=launch,
        experiment_name=name,
        config_hash="abc123",
        params={"seed": "42", "model.params.max_depth": "4"},
        silver_dataset_version_id="silver_x",
        feature_set_id=None,
        split_id="split_y",
        model_type="xgb",
        seed=42,
        artifact_dir="artifacts/runs/" + run_id,
        git_sha="deadbeef",
        git_dirty=False,
    )


def test_run_ids_are_unique_and_prefixed():
    ids = {reg.new_run_id("run") for _ in range(50)}
    assert len(ids) == 50 and all(i.startswith("run_") for i in ids)


def test_launch_and_run_lifecycle(engine):
    launch = reg.start_launch(
        engine, experiment_name="exp", kind="single", n_runs=1, git_sha="x", git_dirty=True
    )
    _start(engine, launch)
    run = reg.get_run(engine, "run_1")
    assert run["status"] == "running" and run["git_dirty"] == 0
    reg.log_metrics(
        engine,
        "run_1",
        [
            {"split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.4},
            {"split": "cv", "fold": 0, "metric": "pr_auc", "value": float("nan")},
        ],
    )
    reg.finish_model_run(
        engine,
        "run_1",
        succeeded=True,
        primary_metric="pr_auc",
        primary_value=0.4,
        n_rounds_final=120,
        n_rows=10_000,
        n_features=6,
        duration_s=3.2,
    )
    reg.finish_launch(engine, launch, succeeded=True)
    run = reg.get_run(engine, "run_1")
    assert run["status"] == "succeeded" and run["primary_value"] == pytest.approx(0.4)
    assert run["n_rounds_final"] == 120 and run["finished_at"]
    assert (run["n_rows"], run["n_features"], run["duration_s"]) == (10_000, 6, pytest.approx(3.2))
    assert reg.get_params(engine, "run_1")["model.params.max_depth"] == "4"
    metrics = {(m["fold"], m["metric"]): m["value"] for m in reg.get_metrics(engine, "run_1")}
    assert metrics[(-1, "pr_auc")] == pytest.approx(0.4) and metrics[(0, "pr_auc")] is None


def test_log_metrics_replaces_same_split_only(engine):
    launch = reg.start_launch(engine, experiment_name="exp", kind="single", n_runs=1)
    _start(engine, launch)
    reg.log_metrics(
        engine, "run_1", [{"split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.1}]
    )
    reg.log_metrics(
        engine, "run_1", [{"split": "holdout", "fold": -1, "metric": "pr_auc", "value": 0.2}]
    )
    reg.log_metrics(
        engine, "run_1", [{"split": "holdout", "fold": -1, "metric": "pr_auc", "value": 0.3}]
    )
    rows = {(m["split"], m["metric"]): m["value"] for m in reg.get_metrics(engine, "run_1")}
    assert rows == {("cv", "pr_auc"): 0.1, ("holdout", "pr_auc"): 0.3}
    assert len(reg.get_metrics(engine, "run_1", split="holdout")) == 1


def test_feature_importance_rows_are_stored_and_replaced(engine):
    launch = reg.start_launch(engine, experiment_name="exp", kind="single", n_runs=1)
    _start(engine, launch)

    def rows(value):
        return [
            {
                "split": "cv",
                "fold": 0,
                "method": "perm",
                "feature": "a",
                "value": value,
                "std": 0.01,
            },
            {
                "split": "cv",
                "fold": -1,
                "method": "perm",
                "feature": "a",
                "value": value,
                "std": None,
            },
            {
                "split": "cv",
                "fold": -1,
                "method": "perm",
                "feature": "b",
                "value": float("nan"),
                "std": float("nan"),
            },
        ]

    reg.log_feature_importance(engine, "run_1", rows(0.2))
    reg.log_feature_importance(engine, "run_1", rows(0.3))  # replaces, no duplicates
    stored = reg.get_feature_importance(engine, "run_1", split="cv", method="perm")
    assert len(stored) == 3
    pooled = {r["feature"]: r for r in stored if r["fold"] == -1}
    assert pooled["a"]["value"] == pytest.approx(0.3) and pooled["a"]["std"] is None
    assert pooled["b"]["value"] is None and pooled["b"]["std"] is None
    assert reg.get_feature_importance(engine, "run_1", method="other") == []


def test_failed_run_and_unknown_ids(engine):
    launch = reg.start_launch(
        engine, experiment_name="exp", kind="sweep", n_runs=2, sweep_path="s.yaml"
    )
    _start(engine, launch)
    reg.finish_model_run(engine, "run_1", succeeded=False, duration_s=0.5, error_summary="boom")
    failed = reg.get_run(engine, "run_1")
    assert failed["error_summary"] == "boom" and failed["duration_s"] == pytest.approx(0.5)
    with pytest.raises(KeyError):
        reg.finish_model_run(engine, "nope", succeeded=True)
    with pytest.raises(KeyError):
        reg.finish_launch(engine, "nope", succeeded=True)
    assert reg.get_run(engine, "nope") is None


def test_list_runs_filters_and_holdout_mark(engine):
    a = reg.start_launch(engine, experiment_name="one", kind="single", n_runs=1)
    b = reg.start_launch(engine, experiment_name="two", kind="single", n_runs=1)
    _start(engine, a, "run_a", "one")
    _start(engine, b, "run_b", "two")
    reg.finish_model_run(engine, "run_a", succeeded=True)
    assert [r["run_id"] for r in reg.list_runs(engine, experiment_name="two")] == ["run_b"]
    assert [r["run_id"] for r in reg.list_runs(engine, status="succeeded")] == ["run_a"]
    assert len(reg.list_runs(engine)) == 2
    assert reg.get_run(engine, "run_a")["holdout_evaluated_at"] is None
    reg.mark_holdout_evaluated(engine, "run_a")
    assert reg.get_run(engine, "run_a")["holdout_evaluated_at"]


def test_nan_stored_as_null(engine):
    assert math.isnan(float("nan"))
    launch = reg.start_launch(engine, experiment_name="exp", kind="single", n_runs=1)
    _start(engine, launch)
    reg.finish_model_run(engine, "run_1", succeeded=True, primary_value=float("nan"))
    assert reg.get_run(engine, "run_1")["primary_value"] is None
