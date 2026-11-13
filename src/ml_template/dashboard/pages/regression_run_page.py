"""Single-run regression panel: no fitting, scoring or analysis action callbacks."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dash import Input, Output, dcc, html

from ml_template.dashboard import regression_data as data
from ml_template.dashboard import regression_figs as figures

P = "regview-"
BINARY = "single-run-binary-content"
REGRESSION = "single-run-regression-content"


def singapore(value):
    if not value:
        return "not recorded"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S SGT")
    except ValueError:
        return str(value)


def table(frame, max_rows=1000):
    if frame is None or frame.empty:
        return html.P("Not recorded / no rows")
    shown = frame.head(max_rows)

    def scalar(value):
        if value is None:
            return "—"
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return "—"
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    return html.Div(
        [
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(str(c)) for c in shown])),
                    html.Tbody(
                        [
                            html.Tr([html.Td(scalar(v)) for v in row])
                            for row in shown.itertuples(index=False, name=None)
                        ]
                    ),
                ],
                style={"width": "100%", "fontSize": "13px"},
            ),
            html.Small(f"Showing {len(shown)} of {len(frame)} rows"),
        ],
        style={"overflowX": "auto"},
    )


def header_view(engine, run_id):
    try:
        h = data.context(engine, run_id)["header"]
        direction = "maximise" if h.get("primary_metric") == "r2" else "minimise"
        return html.Div(
            [
                html.B("Regression | " + str(run_id)),
                html.P(
                    f"Primary: {h.get('primary_metric')} ({direction}) | final rounds: {h.get('n_rounds_final')} | features: {h.get('n_features')}"
                ),
                html.P("Started: " + singapore(h.get("started_at"))),
                html.P("Recorded holdout access: " + singapore(h.get("holdout_evaluated_at"))),
            ]
        )
    except data.ERRORS as exc:
        return html.Pre(f"Regression metadata unavailable: {exc}")


def layout():
    return html.Div(
        [
            html.H3("Regression — saved development artifacts"),
            html.P(
                "Read-only saved development artifacts. Regression comparison/ranking use separate task selectors; Final evaluation displays frozen evidence. No fitting or new evaluation."
            ),
            html.Button("Refresh saved analyses", id=P + "refresh", n_clicks=0),
            html.Label("Choose a saved analysis explicitly; no automatic latest selection"),
            dcc.Dropdown(id=P + "analysis", options=[], value=None, clearable=True),
            html.Label("Permutation-importance fold"),
            dcc.Dropdown(
                id=P + "fold",
                options=[{"label": "Unweighted mean across folds", "value": -1}],
                value=-1,
                clearable=False,
            ),
            html.Div(id=P + "body"),
        ]
    )


def wrap_layout(root, engine, selected_run):
    children = list(root.children)
    boundary = next(
        (i for i, child in enumerate(children) if getattr(child, "id", None) == "run-header")
    )
    regression = data.is_regression_run(engine, selected_run)
    root.children = children[: boundary + 1] + [
        html.Div(
            children[boundary + 1 :], id=BINARY, style={"display": "none"} if regression else {}
        ),
        html.Div(layout(), id=REGRESSION, style={} if regression else {"display": "none"}),
    ]
    return root


def view(engine, run_id, selected=None, fold=-1):
    if not run_id or not data.is_regression_run(engine, run_id):
        return html.P("Select a regression run")
    bundle = data.load_bundle(engine, run_id, selected)
    if bundle["state"] != "verified":
        return html.Pre(bundle.get("message", "No saved data"))
    if (
        isinstance(fold, bool)
        or not isinstance(fold, int)
        or fold not in [-1, *range(bundle["context"]["config"]["split"]["n_folds"])]
    ):
        return html.P("Choose a valid importance fold")
    analysis = bundle["analysis"]
    unit = analysis["summary"]["target_unit"] if analysis else "target units"
    result = [
        html.Ul([html.Li(str(w)) for w in bundle["warnings"]]),
        html.P(
            "Run metadata verified. Selected analysis artifacts are hash-checked. Registry metrics are read, not recomputed; source datasets and model objects are not loaded."
        ),
        html.H4("Development CV metrics"),
        table(figures.metrics_table(bundle["metrics"], unit)),
        dcc.Graph(figure=figures.fold_metrics(bundle["metrics"], unit)),
        html.H4("Recorded inner stopping fits"),
        dcc.Graph(figure=figures.curves(bundle["inner"], "inner", unit)),
        html.H4("Separate outer diagnostic refits"),
        dcc.Graph(figure=figures.curves(bundle["outer"], "outer", unit)),
        html.P(
            "Inner dashed lines record selection. Outer curves do not select rounds and must not be used to retune on the same validation folds."
        ),
    ]
    if analysis is None:
        result.append(
            html.P(
                "Select an analysis for residuals/errors/importance. If absent, generate it from the CLI; the dashboard does not create analyses."
            )
        )
        return html.Div(result)
    if analysis["summary"].get("analysis_kind") == "oof_residuals_v1":
        summary = analysis["summary"]
        pooled = summary["pooled"]
        result.extend(
            [
                html.H4("Development OOF residual summary"),
                html.P(
                    "Residual = observed minus OOF prediction; positive means underprediction. No model replay or holdout labels."
                ),
                table(
                    pd.DataFrame(
                        [
                            {"metric": k, "value": v}
                            for k, v in pooled.items()
                            if k != "residual_quantiles"
                        ]
                    )
                ),
                html.H4("Residual quantiles"),
                table(
                    pd.DataFrame(
                        [
                            {"quantile": k, "residual": v}
                            for k, v in pooled["residual_quantiles"].items()
                        ]
                    )
                ),
                html.H4("Per-fold OOF residual summary"),
                table(pd.DataFrame(summary["fold_summaries"]).drop(columns="residual_quantiles")),
            ]
        )
    result.extend(
        [
            html.H4("Selected analysis: " + analysis["id"]),
            html.P(
                "Recorded units: "
                + str(unit)
                + "; no conversion. Exploratory development OOF results, not new holdout evidence."
            ),
            *[
                dcc.Graph(figure=fig)
                for fig in figures.residual_figures(analysis["residuals"], unit)
            ],
            html.H4("Permutation error increases"),
            html.P(
                "Positive means shuffling hurts. Negative values are retained. Repeat SD and fold SD are different descriptive quantities, not confidence intervals. Missing SD does not mean zero variability."
            ),
            dcc.Graph(figure=figures.importance(analysis["importance"], fold)),
            table(analysis["importance"]),
            html.H4("Saved largest-error examples"),
            table(analysis["errors"]),
        ]
    )
    return html.Div(result)


def register_callbacks(app, engine):

    @app.callback(Output(BINARY, "style"), Output(REGRESSION, "style"), Input("run-dd", "value"))
    def visibility(run_id):
        return (
            ({"display": "none"}, {})
            if data.is_regression_run(engine, run_id)
            else ({}, {"display": "none"})
        )

    @app.callback(
        Output(P + "analysis", "options"),
        Output(P + "analysis", "value"),
        Output(P + "fold", "options"),
        Output(P + "fold", "value"),
        Input("run-dd", "value"),
        Input(P + "refresh", "n_clicks"),
    )
    def choices(run_id, clicks):
        if not run_id or not data.is_regression_run(engine, run_id):
            return ([], None, [{"label": "Mean across folds", "value": -1}], -1)
        try:
            ctx = data.context(engine, run_id)
            options, _, _ = data.discover(ctx)
            folds = [{"label": "Unweighted mean across folds", "value": -1}] + [
                {"label": f"fold {k}", "value": k} for k in range(ctx["config"]["split"]["n_folds"])
            ]
            return (options, None, folds, -1)
        except data.ERRORS:
            return ([], None, [{"label": "Mean across folds", "value": -1}], -1)

    @app.callback(
        Output(P + "body", "children"),
        Input("run-dd", "value"),
        Input(P + "analysis", "value"),
        Input(P + "fold", "value"),
        Input(P + "refresh", "n_clicks"),
    )
    def body(run_id, selected, fold, clicks):
        return view(engine, run_id, selected, -1 if fold is None else fold)
