"""Mandatory real Dash gates. No silent skips; run in the project environment."""

import base64
import io
import zipfile
from types import SimpleNamespace

import pytest
from dash import Dash
from forecasting_dashboard08_fixtures import fixture

from ml_template.dashboard.pages import forecasting_page as page


def test_real_layout_and_callbacks(tmp_path):
    paths = fixture(tmp_path)
    app = Dash(__name__)
    app.layout = page.layout(database=paths["database"])
    page.register_callbacks(app, database=paths["database"])
    assert len(app.callback_map) == 3
    response = app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200 and "forecast-report" in response.get_data(as_text=True)


def test_selection_arity_and_plot_options(tmp_path):
    paths = fixture(tmp_path)
    output = page.selections(paths["database"], paths["identifier"])
    assert len(output) == 11 and len(output[8]) == 2
    assert output[5] == [] and output[7] == []
    assert len(page.selections(paths["database"], None)) == 11


def test_render_and_csv(tmp_path):
    paths = fixture(tmp_path)
    args = [
        paths["database"],
        paths["identifier"],
        "rmse",
        "common_supported",
        ["xgb"],
        "seasonal_naive_7",
        [],
        [],
        ["load"],
    ]
    assert "Current comparison" in str(page.render(*args, '["a"]'))
    result, status = page.download(args[0], 1, *args[1:])
    assert result["filename"].endswith(".csv") and "exported" in status


def test_missing_scores_fallback(tmp_path):
    paths = fixture(tmp_path)
    for file in paths["score_path"].iterdir():
        file.unlink()
    args = [
        paths["database"],
        paths["identifier"],
        "rmse",
        "common_supported",
        ["xgb"],
        "seasonal_naive_7",
        [],
        [],
        ["load"],
    ]
    assert "Current comparison" in str(page.render(*args, '["a"]'))
    args[6] = ['["a"]']
    assert "Filtered aggregates unavailable" in str(page.render(*args, '["a"]'))


def test_app_default_preserved(tmp_path, monkeypatch):
    from dash import html

    from ml_template.dashboard import app as module

    paths = fixture(tmp_path)
    for name in [
        "assets_page",
        "calibration_page",
        "compare_page",
        "evaluation_page",
        "experiment_page",
        "run_page",
        "sweeps_page",
    ]:
        target = getattr(module, name)
        monkeypatch.setattr(target, "layout", lambda engine: html.Div("old page"))
        monkeypatch.setattr(target, "register_callbacks", lambda app, engine: None)
    app = module.create_app(engine=SimpleNamespace(), forecast_database=paths["database"])
    response = app.server.test_client().get("/_dash-layout")
    assert "forecasting-container" in response.get_data(as_text=True)
    assert response.get_json()["props"]["children"][1]["props"]["value"] == "run"


@pytest.mark.parametrize("format_name", ["html", "zip"])
def test_existing_exports(format_name):
    from ml_template.dashboard.report_callbacks import download_snapshot

    snapshot = {
        "schema_version": 1,
        "format": format_name,
        "active_tab": "forecasting",
        "captured_at": "2026-11-16T02:00:00Z",
        "include_final_evaluation": False,
        "controls": [],
        "blocks": [{"kind": "paragraph", "text": "Forecast fixture"}],
    }
    value, status = download_snapshot(snapshot, "forecasting", [])
    assert "prepared" in status
    if format_name == "zip":
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(value["content"]))) as archive:
            assert archive.testzip() is None
    else:
        assert "Forecast fixture" in value["content"]


def test_no_model_execution_in_ui_imports():
    import ast
    from pathlib import Path

    forbidden = ("ml_template.models", "ml_template.training", "ml_template.evaluation")
    for node in ast.walk(ast.parse(Path(page.__file__).read_text())):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(forbidden)
