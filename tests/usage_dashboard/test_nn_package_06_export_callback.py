"""HTTP wiring/side-effect tests for the saved monitoring export callback.

Renderer/verified-loader spies isolate callback wiring. Native monitor/report tests
remain responsible for real artifact verification and ZIP generation.
"""

import base64
import hashlib
import sqlite3

import pytest


@pytest.fixture
def export_ui(tmp_path, monkeypatch):
    dash = pytest.importorskip("dash")
    pytest.importorskip("dash_ag_grid")
    from ml_template.usage_dashboard import monitoring_page as page
    from ml_template.usage_dashboard import nn_monitor_report as report

    database = tmp_path / "project.db"
    with sqlite3.connect(database):
        pass
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    app = dash.Dash(__name__)
    app.layout = lambda: page.monitoring_layout(database)
    page.register_monitoring_callbacks(app, database)
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    calls = []
    view = {"summary": {"monitoring_id": "mon_fixture"}}
    payload = b"PK\x03\x04synthetic callback bytes"

    def load(db, identity):
        calls.append(("load", identity))
        assert db == database and identity == "mon_fixture"
        return view

    def render(value, identity):
        calls.append(("render", identity))
        assert value is view and identity == "mon_fixture"
        return payload

    monkeypatch.setattr(page, "load_assessment", load)
    monkeypatch.setattr(report, "bundle_bytes", render)
    yield (client, calls, payload, page)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["project.db"]


def post_export(client, clicks, identity):
    return client.post(
        "/_dash-update-component",
        json={
            "output": "mon-report-download.data",
            "outputs": {"id": "mon-report-download", "property": "data"},
            "inputs": [{"id": "mon-report-export", "property": "n_clicks", "value": clicks}],
            "state": [{"id": "mon-selection", "property": "value", "value": identity}],
            "changedPropIds": ["mon-report-export.n_clicks"],
        },
    )


@pytest.mark.parametrize(
    "clicks,identity",
    [(0, "mon_fixture"), (1, None)],
)
def test_export_requires_click_and_selection(export_ui, clicks, identity):
    client, calls, _, _ = export_ui

    response = post_export(client, clicks, identity)

    assert response.status_code in (200, 204), response.get_data(as_text=True)

    if response.status_code == 200:
        payload = response.get_json()
        assert isinstance(payload, dict), response.get_data(as_text=True)
        assert payload.get("response") == {}, payload

    assert calls == []


def test_export_consumes_only_selected_saved_view(export_ui):
    client, calls, payload, _ = export_ui
    response = post_export(client, 1, "mon_fixture")
    assert response.status_code == 200
    download = response.get_json()["response"]["mon-report-download"]["data"]
    assert download["filename"] == "mon_fixture.zip" and download["base64"] is True
    assert base64.b64decode(download["content"]) == payload
    assert calls == [("load", "mon_fixture"), ("render", "mon_fixture")]


def test_failed_saved_verification_cannot_download(export_ui, monkeypatch):
    client, calls, _, page = export_ui

    def fail(*args, **kwargs):
        calls.append(("rejected", "mon_fixture"))
        raise ValueError("saved artifact verification failed")

    monkeypatch.setattr(page, "load_assessment", fail)
    response = post_export(client, 1, "mon_fixture")
    assert response.status_code == 500
    assert calls == [("rejected", "mon_fixture")]
