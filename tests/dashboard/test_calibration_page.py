import json
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")
from dash import Dash  # noqa: E402
from plotly.utils import PlotlyJSONEncoder  # noqa: E402

from ml_template.dashboard.pages import calibration_page  # noqa: E402


def component_ids(component):
    output = []
    if hasattr(component, "id"):
        output.append(component.id)
    children = getattr(component, "children", None)
    if children is not None:
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, "to_plotly_json"):
                output.extend(component_ids(child))
    return output


def test_empty_registry_page_is_safe_and_read_only():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA query_only = ON")
    layout = calibration_page.layout(conn)
    ids = component_ids(layout)
    assert len(ids) == len(set(ids))
    assert "calibration-parent" in ids
    assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    json.dumps(layout, cls=PlotlyJSONEncoder)
    conn.close()


def test_callbacks_register_and_layout_endpoint():
    conn = sqlite3.connect(":memory:")
    app = Dash(__name__)
    app.layout = calibration_page.layout(conn)
    calibration_page.register_callbacks(app, conn)
    assert len(app.callback_map) == 3
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    conn.close()


def test_partial_bundle_renders_empty_states():
    bundle = {
        "header": {"calibration_id": "cal_test", "parent_run_id": "run_test"},
        "folder": "/example",
        "config": {},
        "provenance": {},
        "calibrator": {},
        "notes": {},
        "warnings": ["example missing artifact"],
        "inventory": pd.DataFrame(),
        "metrics": pd.DataFrame(),
        "reliability": pd.DataFrame(),
        "predictions": pd.DataFrame(),
    }
    view = calibration_page.build_view(bundle)
    encoded = json.dumps(view, cls=PlotlyJSONEncoder)
    assert "example missing artifact" in encoded
    assert "not final-test" in encoded
