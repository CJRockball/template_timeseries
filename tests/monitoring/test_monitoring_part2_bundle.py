"""Part-two regression tests; run in the installed repository environment."""

import ast
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ml_template.db import scoring_attempt_registry as registry
from ml_template.monitoring import score_pinned
from ml_template.monitoring.scoring_attempts import BatchMetadata, tracked_score


def test_part2_early_failure():
    def operation(*args, **kwargs):
        kwargs["stage_hook"]("release_loading")
        raise ValueError("runtime mismatch")

    with sqlite3.connect(":memory:") as conn:
        with pytest.raises(ValueError):
            tracked_score(operation, conn, "release", "input.csv")
        row = conn.execute(
            "SELECT status,stage,error_type,validation_json FROM scoring_attempts"
        ).fetchone()
        assert row[:3] == ("failed", "release_loading", "ValueError")
        assert json.loads(row[3])["accepted"] is False


def test_part2_success():
    def operation(*args, **kwargs):
        assert kwargs["batch_metadata"]["business_batch_id"] == "business-a"
        kwargs["stage_hook"]("prediction", "score_a")
        return {"scoring_batch_id": "score_a", "status": "succeeded"}

    with sqlite3.connect(":memory:") as conn:
        result = tracked_score(
            operation,
            conn,
            "release",
            "input.csv",
            batch_metadata={"business_batch_id": "business-a"},
        )
        assert result["attempt_id"].startswith("attempt_")
        assert conn.execute("SELECT status,scoring_batch_id FROM scoring_attempts").fetchone() == (
            "succeeded",
            "score_a",
        )


def test_part2_check_writes_nothing():
    def operation(*args, **kwargs):
        assert kwargs["check_only"] and "stage_hook" not in kwargs
        return {"ready": True}

    with sqlite3.connect(":memory:") as conn:
        assert tracked_score(operation, conn, "release", "input.csv", check_only=True)["ready"]
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_part2_invalid_metadata_audited():
    with sqlite3.connect(":memory:") as conn:
        with pytest.raises(ValidationError):
            tracked_score(
                lambda *a, **k: None, conn, "release", "input.csv", batch_metadata={"unexpected": 1}
            )
        assert conn.execute("SELECT status,stage FROM scoring_attempts").fetchone() == (
            "failed",
            "metadata_validation",
        )


def test_part2_immutable_completion():
    with sqlite3.connect(":memory:") as conn:
        registry.start(conn, "a", "r", "input.csv")
        registry.finish(conn, "a", 1.0, {})
        with pytest.raises(ValueError):
            registry.finish(conn, "a", 1.0, {})


@pytest.mark.parametrize(
    "value",
    [
        {"observation_start": "2026-10-22"},
        {"observation_start": "2026-10-22T02:00:00Z", "observation_end": "2026-10-22T01:00:00Z"},
    ],
)
def test_part2_time_contract(value):
    with pytest.raises(ValidationError):
        BatchMetadata(**value)


def test_part2_pinned_symlink_preserved(tmp_path, monkeypatch):
    executable = tmp_path / "python-real"
    executable.touch()
    python = tmp_path / "venv-python"
    python.symlink_to(executable)

    def run(command, **kwargs):
        assert command[0] == str(python) and command[1] == "-I"
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=score_pinned.MARKER + '{"ready":true,"files_written":false}',
        )

    monkeypatch.setattr(score_pinned.subprocess, "run", run)
    assert score_pinned.run_child(python, {"check_only": True})["ready"]


def test_part2_pinned_failure(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setattr(
        score_pinned.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="runtime error", stdout=""),
    )
    with pytest.raises(score_pinned.PinnedScoringError, match="runtime error"):
        score_pinned.run_child(python, {"check_only": False})


def test_part2_no_inference_imports_in_dashboard():
    from ml_template.usage_dashboard import monitoring_page

    root = Path(monitoring_page.__file__).parent
    for name in ("monitoring_data.py", "monitoring_figures.py", "monitoring_page.py"):
        imports = [
            n.module
            for n in ast.walk(ast.parse((root / name).read_text()))
            if isinstance(n, ast.ImportFrom) and n.module
        ]
        assert not any(
            n.startswith(
                (
                    "ml_template.models",
                    "ml_template.training",
                    "ml_template.scoring",
                    "ml_template.dashboard",
                )
            )
            for n in imports
        )


@pytest.mark.skipif(
    importlib.util.find_spec("dash") is None or importlib.util.find_spec("dash_ag_grid") is None,
    reason="requires Dash and AG Grid",
)
def test_part2_ui_is_read_only(tmp_path):
    from dash import Dash

    from ml_template.usage_dashboard.monitoring_page import (
        monitoring_layout,
        register_monitoring_callbacks,
    )

    database = tmp_path / "project.db"
    with sqlite3.connect(database):
        pass
    app = Dash(__name__)
    app.layout = lambda: monitoring_layout(database)
    register_monitoring_callbacks(app, database)
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    assert len(app.callback_map) == 2
    export = app.callback_map["mon-report-download.data"]
    assert export["inputs"] == [{"id": "mon-report-export", "property": "n_clicks"}]
    assert export["state"] == [{"id": "mon-selection", "property": "value"}]
    dependencies = app.server.test_client().get("/_dash-dependencies").get_json()
    download = next(row for row in dependencies if row["output"] == "mon-report-download.data")
    assert download["prevent_initial_call"] is True
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
