"""Optional framework and shared-loader checks plus static import guard."""

import ast
from pathlib import Path

import pytest


def test_regression_dash_callbacks():
    pytest.importorskip("dash")
    from dash import Dash, dcc, html

    from ml_template.dashboard.pages import regression_run_page as page

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Dropdown(id="run-dd"),
            html.Div(id=page.BINARY),
            html.Div(page.layout(), id=page.REGRESSION),
        ]
    )
    page.register_callbacks(app, None)
    assert len(app.callback_map) == 3
    assert page.singapore("2026-11-05T12:27:00+00:00").endswith("20:27:00 SGT")


def test_no_model_or_job_execution_imports():
    from ml_template.dashboard import regression_data, regression_figs

    forbidden = (
        "ml_template.models",
        "ml_template.training",
        "ml_template.evaluation.evaluate_run",
        "ml_template.evaluation.regression_analysis_run",
        "ml_template.db.model_run_registry",
    )
    for module in [regression_data, regression_figs]:
        for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden)


def test_shared_guard_precedes_source_lookup(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from ml_template.dashboard import regression_data, run_data

    def blocked(header):
        raise ValueError("regression guard marker")

    def forbidden(*args, **kwargs):
        raise AssertionError("Source lookup before guard")

    monkeypatch.setattr(regression_data, "guard_binary_labels", blocked)
    monkeypatch.setattr(run_data, "_query", forbidden)
    with pytest.raises(ValueError, match="guard marker"):
        run_data.load_labels(None, {"run_id": "reg"})
