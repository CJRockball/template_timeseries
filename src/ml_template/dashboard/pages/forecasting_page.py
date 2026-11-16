"""Development-only forecasting tab; callbacks read saved evidence only."""

import json
import sqlite3

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ml_template.dashboard import forecasting_data as data
from ml_template.dashboard.forecasting_figs import empty, figures

ERRORS = (ValueError, OSError, sqlite3.Error, KeyError, TypeError, OverflowError)


def database_path(engine=None, database=None):
    if database:
        return str(database)
    if getattr(getattr(engine, "dialect", None), "name", None) == "sqlite":
        return getattr(getattr(engine, "url", None), "database", None)
    return None


def layout(engine=None, database=None):
    options, warnings = data.inventory(database_path(engine, database))
    return html.Div(
        [
            html.H2("Forecasting — Development"),
            html.P(
                "Saved development evidence only. No training, raw Silver reads, final-holdout scoring or promotion."
            ),
            html.P(" | ".join(warnings)),
            html.Label("Acceptance report: one cohort at a time"),
            dcc.Dropdown(id="forecast-report", options=options),
            html.Div(id="forecast-selection-status"),
            html.Label("Methods"),
            dcc.Dropdown(id="forecast-methods", multi=True),
            html.Label("Baseline"),
            dcc.Dropdown(id="forecast-baseline", clearable=False),
            dcc.RadioItems(
                id="forecast-metric", options=["rmse", "mae"], value="rmse", inline=True
            ),
            dcc.RadioItems(
                id="forecast-support",
                options=[
                    {
                        "label": "Common support across ALL report methods",
                        "value": "common_supported",
                    },
                    {"label": "Each method's own support", "value": "own_supported"},
                ],
                value="common_supported",
            ),
            dcc.Checklist(
                id="forecast-detail",
                options=[
                    {
                        "label": "Load linked DEVELOPMENT scores for filtering and diagnostics (200,000 rows maximum)",
                        "value": "load",
                    }
                ],
                value=[],
            ),
            html.Label("Series filter: empty = all; requires linked scores"),
            dcc.Dropdown(id="forecast-series", multi=True),
            html.Label("Origin filter: empty = all; requires linked scores"),
            dcc.Dropdown(id="forecast-origins", multi=True),
            html.Label("Single plot series"),
            dcc.Dropdown(id="forecast-plot-series", clearable=False),
            html.Button("Download current comparison CSV", id="forecast-csv-button", n_clicks=0),
            dcc.Download(id="forecast-csv"),
            html.Div(id="forecast-csv-status"),
            dcc.Loading(html.Div(id="forecast-content")),
        ],
        style={"padding": "16px"},
    )


def selections(database, identifier):
    blank = ([], [], [], None, [], [], [], [], [], None)
    if not identifier:
        return (*blank, "Select a registered acceptance report")
    try:
        state = data.snapshot(database, identifier)
        methods = [{"label": m, "value": m} for m in state["methods"]]
        series = [{"label": key, "value": key} for key in state["series"]]
        origins = [{"label": key, "value": key} for key in state["origins"]]
        baseline = (
            "seasonal_naive_7" if "seasonal_naive_7" in state["methods"] else state["methods"][0]
        )
        return (
            methods,
            state["methods"],
            methods,
            baseline,
            series,
            [],
            origins,
            [],
            series,
            state["series"][0],
            "Frozen report verified; linked scores are checked only when loaded",
        )
    except ERRORS as error:
        return (*blank, "Report unavailable: " + str(error))


def prepare(database, identifier, metric, support, methods, baseline, series, origins, detail):
    state = data.snapshot(database, identifier)
    rows, warning = None, None
    if detail and "load" in detail:
        try:
            rows = data.detail(database, state)
        except ERRORS as error:
            if series or origins:
                raise ValueError("Filtered aggregates unavailable: " + str(error)) from error
            warning = "Summary available, linked diagnostics unavailable: " + str(error)
    result = data.view(state, metric, support, methods, baseline, series or (), origins or (), rows)
    result["warning"] = warning
    return state, result


def table(records, fields):
    def scalar(value):
        if value is None:
            return "unavailable"
        return str(round(value, 6)) if isinstance(value, float) else str(value)

    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(f) for f in fields])),
                html.Tbody(
                    [html.Tr([html.Td(scalar(row.get(f))) for f in fields]) for row in records]
                ),
            ]
        ),
        style={"overflowX": "auto"},
    )


def render(
    database, identifier, metric, support, methods, baseline, series, origins, detail, plot_series
):
    if not identifier:
        return html.P("Select a report; existing dashboard pages remain unchanged")
    try:
        state, result = prepare(
            database, identifier, metric, support, methods, baseline, series, origins, detail
        )
        charts = figures(result, state["methods"], plot_series)
        report = state["report"]
        try:
            settings = data.model_settings(database, state)
        except ERRORS as error:
            settings = {"unavailable": str(error)}
        return html.Div(
            [
                html.P(result["note"]),
                html.P(
                    result["warning"]
                    or "Metrics use the saved target scale; raw errors are not normalized across series"
                ),
                html.H3("Current comparison"),
                table(
                    result["table"],
                    [
                        "method",
                        "n",
                        "total_requests",
                        "coverage_fraction",
                        "mae",
                        "rmse",
                        "bias",
                        "score_minus_baseline",
                        "improvement_percent",
                    ],
                ),
                html.P(
                    "Bias needs linked scores. Percentage improvement is undefined when baseline error is zero."
                ),
                *[
                    dcc.Graph(id="forecast-chart-" + key, figure=charts[key])
                    for key in ["pooled", "origins", "series", "actual", "residuals"]
                ],
                *[html.P(note) for note in charts["notes"]],
                html.H3("Largest supported errors: up to 20 rows"),
                table(
                    data.largest_errors(result),
                    [
                        "series_key",
                        "target_date",
                        "method",
                        "y_true",
                        "y_pred",
                        "error",
                        "absolute_error",
                    ],
                ),
                html.H3("Selected rounds"),
                table(
                    report["rounds"],
                    ["origin", "selected_rounds", "refit_rows", "requests", "supported"],
                ),
                html.Details(
                    [
                        html.Summary("Coverage, exclusions and warnings"),
                        html.Pre(
                            json.dumps(
                                {
                                    key: report[key]
                                    for key in ["unsupported", "sample_eligibility", "warnings"]
                                },
                                indent=2,
                            )
                        ),
                    ]
                ),
                html.Details(
                    [
                        html.Summary("Frozen model settings: no inference"),
                        html.Pre(json.dumps(settings, indent=2)),
                    ]
                ),
                html.Details(
                    [
                        html.Summary("Provenance"),
                        html.Pre(
                            json.dumps(
                                {
                                    "report": state["manifest"]["artifact_id"],
                                    **{
                                        key: report[key]
                                        for key in [
                                            "bindings",
                                            "plan_id",
                                            "silver",
                                            "holdout",
                                            "holdout_scored",
                                        ]
                                    },
                                },
                                indent=2,
                            )
                        ),
                    ]
                ),
                html.Details(
                    [
                        html.Summary("Optional user-associated resource records"),
                        html.Pre(json.dumps(report.get("resources", []), indent=2)),
                    ]
                ),
                html.P(
                    "Top HTML/ZIP buttons export this rendered tab, not a new evaluation or full source artifact."
                ),
            ]
        )
    except ERRORS as error:
        return html.Div(
            [
                html.P("Forecasting view unavailable: " + str(error)),
                dcc.Graph(figure=empty("No stale or partial filtered metrics displayed")),
            ]
        )


def download(
    database, clicks, identifier, metric, support, methods, baseline, series, origins, detail
):
    if not clicks or not identifier:
        raise PreventUpdate
    try:
        _, result = prepare(
            database, identifier, metric, support, methods, baseline, series, origins, detail
        )
        return dcc.send_string(
            data.comparison_csv(result), "forecasting-comparison.csv", type="text/csv"
        ), "Current full-precision comparison exported"
    except ERRORS as error:
        return None, "CSV unavailable: " + str(error)


def register_callbacks(app, engine=None, database=None):
    database = database_path(engine, database)
    app.callback(
        Output("forecast-methods", "options"),
        Output("forecast-methods", "value"),
        Output("forecast-baseline", "options"),
        Output("forecast-baseline", "value"),
        Output("forecast-series", "options"),
        Output("forecast-series", "value"),
        Output("forecast-origins", "options"),
        Output("forecast-origins", "value"),
        Output("forecast-plot-series", "options"),
        Output("forecast-plot-series", "value"),
        Output("forecast-selection-status", "children"),
        Input("forecast-report", "value"),
    )(lambda identifier: selections(database, identifier))
    inputs = [
        Input("forecast-report", "value"),
        Input("forecast-metric", "value"),
        Input("forecast-support", "value"),
        Input("forecast-methods", "value"),
        Input("forecast-baseline", "value"),
        Input("forecast-series", "value"),
        Input("forecast-origins", "value"),
        Input("forecast-detail", "value"),
    ]
    app.callback(
        Output("forecast-content", "children"), *inputs, Input("forecast-plot-series", "value")
    )(
        lambda identifier, metric, support, methods, baseline, series, origins, detail, plot_series: (
            render(
                database,
                identifier,
                metric,
                support,
                methods,
                baseline,
                series,
                origins,
                detail,
                plot_series,
            )
        )
    )
    states = [State(item.component_id, item.component_property) for item in inputs]
    app.callback(
        Output("forecast-csv", "data"),
        Output("forecast-csv-status", "children"),
        Input("forecast-csv-button", "n_clicks"),
        *states,
        prevent_initial_call=True,
    )(
        lambda clicks, identifier, metric, support, methods, baseline, series, origins, detail: (
            download(
                database,
                clicks,
                identifier,
                metric,
                support,
                methods,
                baseline,
                series,
                origins,
                detail,
            )
        )
    )
