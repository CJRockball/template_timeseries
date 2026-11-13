import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from ml_template.db import tracking_tables as tt  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    return sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def _columns(engine, table):
    return {c["name"] for c in sqlalchemy.inspect(engine).get_columns(table)}


def test_creates_all_tables(engine):
    tt.ensure_tracking_tables(engine)
    names = set(sqlalchemy.inspect(engine).get_table_names())
    assert {
        "splits",
        "experiment_launches",
        "model_runs",
        "run_params",
        "run_metrics",
        "run_feature_importance",
    } <= names
    assert {"n_rows", "n_features", "duration_s"} <= _columns(engine, "model_runs")


def test_is_idempotent(engine):
    tt.ensure_tracking_tables(engine)
    tt.ensure_tracking_tables(engine)
    assert "duration_s" in _columns(engine, "model_runs")


def test_older_database_gains_new_nullable_columns_and_keeps_rows(engine):
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE experiment_launches (experiment_run_id TEXT PRIMARY KEY, "
                "experiment_name TEXT NOT NULL, kind TEXT NOT NULL, sweep_path TEXT, "
                "n_runs INTEGER NOT NULL, n_failed INTEGER, status TEXT NOT NULL, "
                "started_at TEXT NOT NULL, finished_at TEXT, git_sha TEXT, git_dirty INTEGER, "
                "error_summary TEXT)"
            )
        )
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE model_runs (run_id TEXT PRIMARY KEY, experiment_run_id TEXT NOT NULL, "
                "experiment_name TEXT NOT NULL, config_hash TEXT NOT NULL, status TEXT NOT NULL, "
                "started_at TEXT NOT NULL, finished_at TEXT, silver_dataset_version_id TEXT NOT NULL, "
                "feature_set_id TEXT, split_id TEXT NOT NULL, model_type TEXT NOT NULL, seed INTEGER, "
                "git_sha TEXT, git_dirty INTEGER, primary_metric TEXT, primary_value FLOAT, "
                "n_rounds_final INTEGER, artifact_dir TEXT, holdout_evaluated_at TEXT, "
                "error_summary TEXT)"
            )
        )
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO model_runs (run_id, experiment_run_id, experiment_name, config_hash, "
                "status, started_at, silver_dataset_version_id, split_id, model_type) "
                "VALUES ('old', 'e', 'x', 'h', 'succeeded', 't', 's', 'sp', 'xgb')"
            )
        )
    assert "duration_s" not in _columns(engine, "model_runs")
    tt.ensure_tracking_tables(engine)
    assert {"n_rows", "n_features", "duration_s"} <= _columns(engine, "model_runs")
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.text("SELECT run_id, duration_s FROM model_runs")).all()
    assert row == [("old", None)]
