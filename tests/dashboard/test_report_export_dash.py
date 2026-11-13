"""Real Dash helper/registration tests; must execute in the user's environment."""

import base64
import io
import zipfile

import pytest


def snap():
    return {
        "schema_version": 1,
        "format": "html",
        "active_tab": "experiments",
        "captured_at": "2026-11-11T14:39:00Z",
        "include_final_evaluation": False,
        "controls": [],
        "blocks": [{"kind": "paragraph", "text": "Synthetic evidence"}],
    }


def test_real_dash_registration():
    pytest.importorskip("dash")
    from dash import Dash, dcc, html

    from ml_template.dashboard.report_callbacks import export_toolbar, register_exports

    app = Dash(__name__)
    app.layout = html.Div([export_toolbar(), dcc.Tabs(id="dashboard-tab", value="experiments")])
    register_exports(app)
    assert len(app.callback_map) == 2
    assert len(app._inline_scripts) == 1


@pytest.mark.parametrize("format", ["html", "zip"])
def test_real_download_helpers(format):
    pytest.importorskip("dash")
    from ml_template.dashboard.report_callbacks import download_snapshot

    value = snap()
    value["format"] = format
    data, status = download_snapshot(value, "experiments", [])
    assert data["filename"].endswith("." + format) and "prepared" in status
    if format == "html":
        assert "Synthetic evidence" in data["content"]
    else:
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(data["content"]))) as archive:
            assert archive.testzip() is None


def test_tab_and_consent_guards():
    pytest.importorskip("dash")
    from dash import no_update

    from ml_template.dashboard.report_callbacks import download_snapshot

    value = snap()
    result, _ = download_snapshot(value, "run", [])
    assert result is no_update
    value["active_tab"] = "evaluation"
    value["include_final_evaluation"] = True
    result, _ = download_snapshot(value, "evaluation", [])
    assert result is no_update


def test_malformed_snapshot():
    pytest.importorskip("dash")
    from dash import no_update

    from ml_template.dashboard.report_callbacks import download_snapshot

    result, message = download_snapshot([1], "experiments", [])
    assert result is no_update and "unavailable" in message
