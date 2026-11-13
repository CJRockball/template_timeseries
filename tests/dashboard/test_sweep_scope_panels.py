"""Real Dash checks for whole-launch callback scope; run in the project environment."""

import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")

from dash import Dash, html  # noqa: E402

from ml_template.dashboard import sweep_parameter_panels as panels  # noqa: E402


def test_callback_ignores_diagnostics_cohort_and_uses_yaml_loader(monkeypatch):
    frame = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "experiment_run_id": ["launch"] * 3,
            "status": ["succeeded"] * 3,
            "primary_value": [0.9, 0.8, 0.7],
            "silver_dataset_version_id": ["silver"] * 3,
            "split_id": ["split"] * 3,
            "primary_metric": ["auc"] * 3,
            "param::model.params.learning_rate": ["0.03", "0.1", "0.03"],
        }
    )
    spec = {"base": "not-read.yaml", "grid": {"model.params.learning_rate": [0.03, 0.1]}}
    monkeypatch.setattr(
        panels, "load_launch_summary_inputs", lambda engine, launch: (frame, spec, "sweep.yaml")
    )
    app = Dash(__name__)
    app.layout = html.Div(panels.parameter_panels("scope-"))
    panels.register_parameter_callbacks(app, None, None, "scope-")
    callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
    overall, top, note, title = callback("launch", "not-even-cohort-json", "higher", 10)
    assert "Top-3 of 3" in title
    assert len(overall.rowData) == 2 and len(top.rowData) == 1
    assert "sweep.yaml" in note and "whole experiment_run_id" in note


def test_missing_yaml_shows_error_not_all_config_parameters(monkeypatch):
    def missing(engine, launch):
        raise ValueError("Recorded sweep YAML not found")

    monkeypatch.setattr(panels, "load_launch_summary_inputs", missing)
    app = Dash(__name__)
    app.layout = html.Div(panels.parameter_panels("scope-"))
    panels.register_parameter_callbacks(app, None, None, "scope-")
    callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
    overall, top, note, title = callback("launch", None, "higher", 10)
    assert "not found" in overall.children
    assert "not found" in top.children
    assert "unavailable" in note
