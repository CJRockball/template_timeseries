"""Click-only snapshot downloads; no registry, source data or model APIs."""

from datetime import UTC, datetime
from uuid import uuid4

from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from ml_template.dashboard.report_client import CLIENTSIDE_CAPTURE
from ml_template.dashboard.report_export import build_bundle, build_html


def export_toolbar():
    return html.Div(
        [
            html.Button("Save current tab — HTML", id="report-export-html", n_clicks=0),
            html.Button(
                "Save current tab — ZIP bundle",
                id="report-export-zip",
                n_clicks=0,
                style={"marginLeft": "8px"},
            ),
            dcc.Checklist(
                id="report-export-consent",
                value=[],
                options=[
                    {
                        "label": "Allow exporting visible Final evaluation content",
                        "value": "allow_final_evaluation",
                    }
                ],
            ),
            html.Small(
                "Rendered tab only. Tables keep displayed precision/limits; figures include chart data. No source files/models are loaded."
            ),
            html.Div(id="report-export-status", role="status", **{"aria-live": "polite"}),
            dcc.Store(id="report-export-snapshot", storage_type="memory"),
            dcc.Download(id="report-export-download"),
        ],
        style={"padding": "12px", "borderBottom": "1px solid #ddd"},
    )


def download_snapshot(snapshot, current_tab, consent):
    if not snapshot:
        raise PreventUpdate
    try:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid browser snapshot")
        if snapshot.get("error"):
            raise ValueError(str(snapshot["error"])[:1500])
        if snapshot.get("active_tab") != current_tab:
            raise ValueError("Active tab changed; click export again")
        if current_tab == "evaluation" and not (
            isinstance(consent, list) and "allow_final_evaluation" in consent
        ):
            raise ValueError("Explicitly allow visible Final evaluation content first")
        name = (
            "dashboard-"
            + str(current_tab)
            + "-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
        if snapshot.get("format") == "html":
            result = dcc.send_string(
                build_html(snapshot), name + ".html", type="text/html;charset=utf-8"
            )
        elif snapshot.get("format") == "zip":
            result = dcc.send_bytes(build_bundle(snapshot), name + ".zip", type="application/zip")
        else:
            raise ValueError("Unknown report format")
        return result, "Download prepared: rendered snapshot, not full source artifacts."
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        return no_update, "Export unavailable: " + str(exc)[:1500]


def register_exports(app):
    app.clientside_callback(
        CLIENTSIDE_CAPTURE,
        Output("report-export-snapshot", "data"),
        Input("report-export-html", "n_clicks"),
        Input("report-export-zip", "n_clicks"),
        State("dashboard-tab", "value"),
        State("report-export-consent", "value"),
        prevent_initial_call=True,
    )
    app.callback(
        Output("report-export-download", "data"),
        Output("report-export-status", "children"),
        Input("report-export-snapshot", "data"),
        State("dashboard-tab", "value"),
        State("report-export-consent", "value"),
        prevent_initial_call=True,
    )(download_snapshot)
