"""Read-only monitoring tab. No database creation, migrations, scoring or fitting."""

import json
import sqlite3

from ml_template.usage_dashboard.monitoring_data import (
    comparable_history,
    inventory,
    load_assessment,
    metric_rows,
    policy_rows,
)
from ml_template.usage_dashboard.monitoring_figures import (
    drift_figure,
    empty,
    history_figure,
    reliability_figure,
)


def monitoring_payload(database, identity):
    if not identity:
        return (
            "No successful monitoring assessment selected",
            [],
            [],
            [],
            [],
            empty("No assessment"),
            empty("No labels"),
            empty("No history"),
            "",
            "",
        )
    try:
        view = load_assessment(database, identity)
        summary, perf = view["summary"], view["performance"]
        rows, errors = comparable_history(database, summary)
        findings = [
            {**row, "evidence": json.dumps(row.get("evidence", {}), sort_keys=True)}
            for row in view["findings"]
        ]
        notes = [
            "Saved artifacts only. No warning is not proof of model health.",
            "Inventories/history are bounded to the latest 200 executions, not a complete audit.",
            "History: latest successful assessment per batch within the same release/reference/outcome-definition/sampling context.",
            "Overlapping batches are not independent replications. Self-reference is a plumbing check.",
        ]
        if summary.get("policy_kind") == "none":
            notes.append(
                "No frozen action policy: flagged/confusion/precision/recall metrics are unavailable."
            )
        if perf is None:
            notes.append("No label import selected. Dataset labels are not loaded automatically.")
        else:
            notes.append(
                "Outcome metrics describe the mature labelled subset, not automatically the full population."
            )
        if summary.get("task") == "regression":
            notes.append(
                "Regression bias = prediction minus truth; residual = truth minus prediction. No probability calibration curve applies."
            )
        if summary.get("model_type") == "nn":
            notes.append(
                "NN assessment scope: " + json.dumps(summary["assessment_scope"], sort_keys=True)
            )
            notes.append("Verified Package 05 pipeline: " + summary["nn_context"]["pipeline_id"])
        if errors:
            notes.append("History verification failures: " + " | ".join(errors))
        details = {
            "coverage_and_support": None
            if perf is None
            else {
                k: perf.get(k)
                for k in (
                    "n_scored",
                    "n_mature",
                    "n_known_as_of",
                    "n_evaluated",
                    "n_positive",
                    "n_negative",
                    "mature_label_coverage",
                    "sufficient_support",
                )
            },
            "configuration": view["config"],
            "distribution_details": view["drift"],
        }
        return (
            json.dumps(summary, indent=2),
            findings,
            [
                {
                    **row,
                    "target_unit": "dimensionless"
                    if row.get("metric") in {"r2", "nominal_coverage", "empirical_coverage"}
                    else summary.get("target_unit"),
                }
                for row in metric_rows(perf)
            ],
            policy_rows(perf),
            view["quality"].get("features", []),
            drift_figure(view["drift"]),
            reliability_figure(perf),
            history_figure(rows),
            "\n".join(notes),
            json.dumps(details, indent=2),
        )
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        msg = f"Monitoring result unavailable or failed verification: {exc}"
        return (msg, [], [], [], [], empty(msg), empty("Unavailable"), empty("Unavailable"), "", "")


def monitoring_layout(database):
    import dash_ag_grid as dag
    from dash import dcc, html

    error = ""
    try:
        runs = inventory(database, "monitoring_runs")
        batches = inventory(database, "scoring_batches")
        attempts = inventory(database, "scoring_attempts")
    except (ValueError, OSError, sqlite3.Error) as exc:
        runs, batches, attempts = [], [], []
        error = str(exc)
    options = [
        {
            "label": f"{r['monitoring_id']} | {r['release_id']} | {r['scoring_batch_id']} | {r['assessment']}",
            "value": r["monitoring_id"],
        }
        for r in runs
        if r["status"] == "succeeded"
    ]

    def grid(identity, columns, rows=None):
        return dag.AgGrid(
            id=identity,
            rowData=[{name: r.get(name) for name in columns} for r in (rows or [])],
            columnDefs=[{"field": name} for name in columns],
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 10,
                "suppressFieldDotNotation": True,
            },
            style={"height": "310px"},
        )

    return html.Div(
        [
            html.H1("Operational tracking and monitoring"),
            html.P(
                "Read-only. Refresh to discover new executions. No automatic retraining. Times retain explicit timezone."
            ),
            html.P(
                "At most 200 records per inventory. Historical batches may have no attempt record."
            ),
            html.Pre(error),
            html.H2("Scoring attempts"),
            grid(
                "mon-attempts",
                [
                    "attempt_id",
                    "status",
                    "stage",
                    "scoring_batch_id",
                    "started_at",
                    "duration_s",
                    "error_type",
                    "error_summary",
                    "metadata_json",
                    "validation_json",
                    "timings_json",
                ],
                attempts,
            ),
            html.H2("Scoring batches"),
            grid(
                "mon-batches",
                [
                    "scoring_batch_id",
                    "release_id",
                    "status",
                    "started_at",
                    "n_rows",
                    "n_flagged",
                    "duration_s",
                    "error_summary",
                ],
                batches,
            ),
            html.H2("Monitoring executions"),
            grid(
                "mon-executions",
                [
                    "monitoring_id",
                    "scoring_batch_id",
                    "release_id",
                    "status",
                    "assessment",
                    "started_at",
                    "label_import_id",
                    "error_summary",
                ],
                runs,
            ),
            dcc.Dropdown(
                id="mon-selection",
                options=options,
                value=options[0]["value"] if options else None,
                clearable=False,
            ),
            html.Button(
                "Export selected saved monitoring report — ZIP", id="mon-report-export", n_clicks=0
            ),
            dcc.Download(id="mon-report-download"),
            html.Pre(id="mon-summary"),
            html.Pre(id="mon-notes", style={"whiteSpace": "pre-wrap"}),
            html.H2("Findings"),
            grid("mon-findings", ["rule", "severity", "message", "action", "evidence"]),
            dcc.Graph(id="mon-drift"),
            html.H2("Input quality"),
            grid(
                "mon-quality", ["feature", "n_rows", "missing", "outside_range", "lower", "upper"]
            ),
            html.H2("Labelled performance"),
            grid(
                "mon-metrics",
                [
                    "variant",
                    "metric",
                    "value",
                    "n_evaluated",
                    "n_positive",
                    "n_negative",
                    "sufficient_support",
                    "target_unit",
                    "level",
                ],
            ),
            dcc.Graph(id="mon-reliability"),
            html.H2("Original saved policy decisions"),
            grid("mon-policy", ["metric", "value"]),
            dcc.Graph(id="mon-history"),
            html.Details(
                [
                    html.Summary("Coverage, configuration and distribution counts"),
                    html.Pre(id="mon-details"),
                ]
            ),
        ],
        style={"maxWidth": "1500px", "margin": "2rem auto", "fontFamily": "system-ui"},
    )


def register_monitoring_callbacks(app, database):
    from dash import Input, Output

    @app.callback(
        Output("mon-summary", "children"),
        Output("mon-findings", "rowData"),
        Output("mon-metrics", "rowData"),
        Output("mon-policy", "rowData"),
        Output("mon-quality", "rowData"),
        Output("mon-drift", "figure"),
        Output("mon-reliability", "figure"),
        Output("mon-history", "figure"),
        Output("mon-notes", "children"),
        Output("mon-details", "children"),
        Input("mon-selection", "value"),
    )
    def select_assessment(identity):
        return monitoring_payload(database, identity)

    from dash import State, dcc, no_update

    @app.callback(
        Output("mon-report-download", "data"),
        Input("mon-report-export", "n_clicks"),
        State("mon-selection", "value"),
        prevent_initial_call=True,
    )
    def export_selected(clicks, identity):
        if not clicks or not identity:
            return no_update
        from ml_template.usage_dashboard.nn_monitor_report import bundle_bytes

        payload = bundle_bytes(load_assessment(database, identity), identity)
        return dcc.send_bytes(lambda stream: stream.write(payload), identity + ".zip")
