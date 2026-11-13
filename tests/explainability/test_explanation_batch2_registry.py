import sqlite3

import pytest

from ml_template.db import explanation_registry as registry


def start(database):
    registry.start(
        database,
        "explain_fixture",
        {},
        {
            "release_id": "rel_fixture",
            "scoring_batch_id": "score_fixture",
            "binding_sha256": "hash",
        },
        "/artifacts/explain_fixture",
    )


def test_registry_success_is_additive(tmp_path):
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE existing_data (value TEXT)")
        conn.execute("INSERT INTO existing_data VALUES ('preserve')")
    start(database)
    registry.finish(
        database,
        "explain_fixture",
        duration_s=1,
        manifest={"content_sha256": "output"},
        n_rows=3,
        n_features=2,
    )
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT value FROM existing_data").fetchone() == ("preserve",)
        assert conn.execute(
            "SELECT status,n_rows,n_features,content_sha256 FROM explanation_runs"
        ).fetchone() == ("succeeded", 3, 2, "output")
    with pytest.raises(ValueError, match="running"):
        registry.finish(database, "explain_fixture", duration_s=2)


def test_registry_failure_keeps_error_and_path(tmp_path):
    database = tmp_path / "project.db"
    sqlite3.connect(database).close()
    start(database)
    registry.finish(
        database, "explain_fixture", duration_s=1, error="bad", artifact_dir="/failed/run"
    )
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT status,error_summary,artifact_dir FROM explanation_runs"
        ).fetchone() == ("failed", "bad", "/failed/run")


def test_missing_database_not_created(tmp_path):
    database = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="existing"):
        start(database)
    assert not database.exists()


def test_duplicate_execution_rejected(tmp_path):
    database = tmp_path / "project.db"
    sqlite3.connect(database).close()
    start(database)
    with pytest.raises(sqlite3.IntegrityError):
        start(database)
