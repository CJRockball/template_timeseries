"""Mandatory real Dash regression tests for forecasting-only DB startup."""

import sqlite3

import pytest

from ml_template.dashboard.forecasting_app import create_app


def test_empty_db_without_model_runs_starts_read_only(tmp_path):
    database = tmp_path / "forecast.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE temporal_acceptance_artifacts(artifact_id TEXT PRIMARY KEY,kind TEXT,manifest_sha256 TEXT,path TEXT)"
        )
    before = database.read_bytes()
    app = create_app(database)
    response = app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200
    assert "forecasting-container" in response.get_data(as_text=True)
    assert database.read_bytes() == before
    with sqlite3.connect(database) as conn:
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='model_runs'").fetchone()
            is None
        )


def test_never_calls_legacy_engine_or_layout(tmp_path, monkeypatch):
    from ml_template.dashboard.pages import assets_page
    from ml_template.db import connection

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy startup path must not run")

    monkeypatch.setattr(connection, "get_engine", forbidden)
    monkeypatch.setattr(assets_page, "layout", forbidden)
    database = tmp_path / "forecast.sqlite"
    database.touch()
    app = create_app(database)
    assert app.server.test_client().get("/_dash-layout").status_code == 200


def test_missing_database_not_created(tmp_path):
    database = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError):
        create_app(database)
    assert not database.exists()


def test_forecasting_is_only_tab_and_exports_registered(tmp_path):
    database = tmp_path / "forecast.sqlite"
    database.touch()
    app = create_app(database)
    layout = app.server.test_client().get("/_dash-layout").get_json()
    tabs = layout["props"]["children"][1]["props"]
    assert tabs["value"] == "forecasting" and len(tabs["children"]) == 1
    assert len(app.callback_map) == 5
