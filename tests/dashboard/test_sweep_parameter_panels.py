"""Real Dash smoke tests; run inside the project's installed environment."""

import json

import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")

from dash import Dash, html  # noqa: E402

from ml_template.dashboard.sweep_parameter_panels import (  # noqa: E402
    parameter_panels,
    register_parameter_callbacks,
    summary_content,
)


def data():
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "status": ["succeeded"] * 3,
            "primary_value": [0.9, 0.8, 0.7],
            "silver_dataset_version_id": ["silver"] * 3,
            "split_id": ["split"] * 3,
            "primary_metric": ["auc"] * 3,
            "param::model.params.learning_rate": ["0.03", "0.03", "0.1"],
        }
    )


def test_real_grid_serialization():
    overall, top, note, title = summary_content(data())
    assert len(overall.rowData) == 2
    assert top.rowData[0]["mode_share_pct"] == pytest.approx(200 / 3)
    assert "Top-3 of 3" in title
    assert "3 successful" in note
    from plotly.utils import PlotlyJSONEncoder

    singleton = next(row for row in overall.rowData if row["n_runs"] == 1)
    assert singleton["std"] is None

    json.dumps(overall.to_plotly_json(), cls=PlotlyJSONEncoder, allow_nan=False)
    json.dumps(top.to_plotly_json(), cls=PlotlyJSONEncoder, allow_nan=False)


def test_one_run_and_empty_states():
    assert "Insufficient runs" in summary_content(data().iloc[:1])[1].children
    assert "No successful" in summary_content(data().iloc[:0])[0].children


def test_callback_registration_and_execution(monkeypatch):
    monkeypatch.setattr(
        "ml_template.dashboard.sweep_parameter_panels.load_launch_summary_inputs",
        lambda engine, launch: (
            data(),
            {"base": "unused.yaml", "grid": {"model.params.learning_rate": [0.03, 0.1]}},
            "synthetic sweep YAML",
        ),
    )
    app = Dash(__name__)
    app.layout = html.Div(parameter_panels("test-"))
    register_parameter_callbacks(app, None, lambda engine, launch: data(), "test-")
    assert len(app.callback_map) == 1
    callback = next(iter(app.callback_map.values()))["callback"].__wrapped__
    result = callback("launch", json.dumps(["silver", "split", "auc"]), "higher", 10)
    assert "Top-3 of 3" in result[3]


def test_panels_have_unique_ids_and_default_ten():
    panels = parameter_panels("test-")
    ids = [component.id for component in panels if getattr(component, "id", None)]
    assert len(ids) == len(set(ids))
    selector = next(
        component for component in panels if getattr(component, "id", None) == "test-parameter-k"
    )
    assert selector.value == 10
