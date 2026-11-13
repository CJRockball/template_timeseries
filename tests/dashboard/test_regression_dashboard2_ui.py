"""Optional Dash wiring and no-execution import checks."""

import ast
from pathlib import Path

import pytest


def test_no_execution_imports():
    from ml_template.dashboard import regression_multi_data

    forbidden = (
        "ml_template.models",
        "ml_template.training",
        "ml_template.evaluation.evaluate_run",
        "ml_template.evaluation.regression_analysis_run",
        "ml_template.db.model_run_registry",
    )
    for path in [
        Path(regression_multi_data.__file__),
        Path(regression_multi_data.__file__).with_name("regression_multi_views.py"),
    ]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden)


def test_dash_panels_build_without_backend_execution():
    pytest.importorskip("dash")
    from dash import Dash, dcc, html

    from ml_template.dashboard import regression_multi_views as view

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Dropdown(id="final-eval-id"),
            view.evaluation_controls(),
            view.wrap(html.Div(), None, "compare"),
            view.wrap(html.Div(), None, "sweeps"),
            view.wrap(html.Div(), None, "experiments"),
        ]
    )
    view.register_compare(app, None)
    view.register_ranking(app, None, "sweeps")
    view.register_ranking(app, None, "experiments")
    view.register_evaluation(app, None)
    assert len(app.callback_map) >= 10
