# threshold-diagnostics-v1
"""Single-run page: layout, view builders and callbacks.

View builders take an engine and a run id and return plain data or components, so they
can be tested without a browser. ``register_callbacks`` only wires them to the layout.
"""

from __future__ import annotations

from functools import lru_cache

import dash_ag_grid as dag
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard import regression_data as _reg_data
from ml_template.dashboard import run_data as rd
from ml_template.dashboard import run_figs as rf
from ml_template.dashboard import threshold_panels
from ml_template.dashboard.pages import regression_run_page as _reg_page

ALL = "(all)"
SECTION = {"marginTop": "32px"}
CARD = {"padding": "6px 12px", "background": "#f3f4f6", "borderRadius": "6px"}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _grid(df: pd.DataFrame, height: int = 280) -> dag.AgGrid:
    shown = df.round(4)
    return dag.AgGrid(
        rowData=shown.to_dict("records"),
        columnDefs=[{"field": str(c)} for c in shown.columns],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        columnSize="sizeToFit",
        style={"height": f"{height}px"},
    )


@lru_cache(maxsize=8)
def _scored(engine: Engine, run_id: str) -> pd.DataFrame | None:
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return None
    try:
        header = rd.run_header(engine, run_id)
        return rd.load_oof_with_labels(engine, header, rd.run_dir(header))
    except (ValueError, KeyError, OSError):
        return None


def run_choices(engine: Engine, experiment: str | None) -> tuple[list[dict], str | None]:
    """Run dropdown options, newest first, and the default (latest) run."""
    runs = rd.list_runs(engine, None if experiment in (None, ALL) else experiment)
    options = [
        {
            "label": f"{r.run_id} | seed {r.seed} | {_fmt(r.primary_value)}",
            "value": r.run_id,
        }
        for r in runs.itertuples()
    ]
    return options, (options[0]["value"] if options else None)


def holdout_label(evaluated_at) -> str:
    if evaluated_at is None or (isinstance(evaluated_at, float) and pd.isna(evaluated_at)):
        return "holdout untouched"
    return f"holdout looked at {evaluated_at}"


def header_view(engine: Engine, run_id: str | None) -> html.Div:
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return _reg_page.header_view(engine, run_id)
    if not run_id:
        return html.Div("No runs found. Train one with `make train`.")
    h = rd.run_header(engine, run_id)
    items = [
        ("experiment", h["experiment_name"]),
        ("feature set", h["feature_set_id"] or "raw features"),
        ("seed", h["seed"]),
        (h["primary_metric"] or "primary", _fmt(h["primary_value"])),
        ("rounds", _fmt(h["n_rounds_final"], 0)),
        ("features", _fmt(h["n_features"], 0)),
        ("duration (s)", _fmt(h["duration_s"], 1)),
        ("started (UTC)", h["started_at"]),
    ]
    badge = holdout_label(h["holdout_evaluated_at"])
    cards = [html.Div([html.Small(f"{k}: "), html.B(_fmt(v))], style=CARD) for k, v in items]
    cards.append(html.Div(html.B(badge), style={**CARD, "background": "#fde68a"}))
    return html.Div(cards, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"})


def metrics_view(engine: Engine, run_id: str | None):
    """Metric dropdown options, default metric, and a summary table component."""
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return ([], None, html.P("Regression metrics are shown below"))
    if not run_id:
        return ([], None, html.Div("No run selected"))
    m = rd.load_metrics(engine, run_id)
    if m.empty:
        return ([], None, html.Div("No metrics stored for this run"))
    cv = m[m["split"] == "cv"]
    table = pd.DataFrame({"pooled": cv[cv["fold"] == -1].set_index("metric")["value"]})
    per_fold = cv[cv["fold"] >= 0].groupby("metric")["value"].agg(["mean", "std"])
    table = table.join(per_fold, how="outer")
    hold = m[(m["split"] == "holdout") & (m["fold"] == -1)].set_index("metric")["value"]
    if not hold.empty:
        table["holdout"] = hold
    table = table.rename(columns={"mean": "fold mean", "std": "fold std"})
    table = table.reset_index(names="metric")
    metrics = sorted(table["metric"])
    primary = rd.run_header(engine, run_id)["primary_metric"]
    return (metrics, primary if primary in metrics else metrics[0], _grid(table))


def curves_view(engine: Engine, run_id: str | None, overlay: bool):
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return (
            rf.empty_fig("Use the regression panel"),
            rf.empty_fig("Inner and outer fits are separate"),
        )
    table = None
    if run_id:
        table = rd.load_learning_curves(rd.run_dir(rd.run_header(engine, run_id)))
    return (rf.learning_curve_fig(table, overlay), rf.gap_fig(table))


def importance_methods(engine: Engine, run_id: str | None) -> tuple[list[str], str | None]:
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return ([], None)
    if not run_id:
        return ([], None)
    methods = sorted(rd.load_importance(engine, run_id)["method"].unique())
    return (methods, methods[0] if methods else None)


def importance_view(engine: Engine, run_id: str | None, method: str | None):
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return rf.empty_fig("Select a saved regression analysis below")
    imp = rd.load_importance(engine, run_id) if run_id else None
    return rf.importance_fig(imp, method)


def scoring_figs(engine: Engine, run_id: str | None):
    scored = _scored(engine, run_id) if run_id else None
    if scored is None:
        msg = "Needs oof_predictions.parquet and the Silver labels for this run"
        return rf.empty_fig(msg), rf.empty_fig(msg)
    y, p = scored["y_true"].to_numpy(), scored["p_oof"].to_numpy()
    return rf.pr_curve_fig(y, p), rf.reliability_fig(y, p)


def threshold_view(engine: Engine, run_id: str | None, threshold: float):
    scored = _scored(engine, run_id) if run_id else None
    if scored is None:
        return rf.empty_fig("No labelled predictions for this run"), "-"
    y, p = scored["y_true"].to_numpy(), scored["p_oof"].to_numpy()
    s = rf.threshold_summary(y, p, threshold)
    text = (
        f"cutoff {threshold:.2f}: flagged {s['alert_rate']:.1%} of rows | "
        f"precision {_fmt(s['precision'], 3)} | recall {_fmt(s['recall'], 3)} | "
        f"TP {s['tp']}  FP {s['fp']}  FN {s['fn']}  TN {s['tn']}"
    )
    return rf.score_hist_fig(y, p, threshold), text


def errors_view(engine: Engine, run_id: str | None):
    if run_id and _reg_data.is_regression_run(engine, run_id):
        return html.P("Saved regression error examples are shown below")
    if not run_id:
        return html.Div("No run selected")
    sample = rd.load_error_sample(rd.run_dir(rd.run_header(engine, run_id)))
    if sample is None:
        return html.Div("No error sample for this run (analysis.error_sample_size is 0)")
    return _grid(sample, height=360)


def cost_view(engine: Engine, run_id: str | None):
    if not run_id:
        return rf.empty_fig("No run selected"), "-"
    h = rd.run_header(engine, run_id)
    runs = rd.list_runs(engine, h["experiment_name"])
    per_round = (
        h["duration_s"] / h["n_rounds_final"]
        if h["n_rounds_final"] and not pd.isna(h["duration_s"])
        else float("nan")
    )
    text = (
        f"duration {_fmt(h['duration_s'], 1)} s | rounds {_fmt(h['n_rounds_final'], 0)} | "
        f"{_fmt(per_round, 3)} s per round | experiment median "
        f"{_fmt(float(runs['duration_s'].median()), 1)} s over {len(runs)} runs"
    )
    return rf.cost_fig(runs, run_id), text


def layout(engine: Engine) -> html.Div:
    runs = rd.list_runs(engine)
    experiments = [ALL, *sorted(runs["experiment_name"].unique())]
    options, default = run_choices(engine, ALL)
    return _reg_page.wrap_layout(
        html.Div(
            [
                html.H2("Single run"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Experiment"),
                                dcc.Dropdown(experiments, ALL, id="experiment-dd", clearable=False),
                            ],
                            style={"width": "280px"},
                        ),
                        html.Div(
                            [
                                html.Label("Run (latest first)"),
                                dcc.Dropdown(options, default, id="run-dd", clearable=False),
                            ],
                            style={"width": "520px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "16px"},
                ),
                html.Div(id="run-header", style={"marginTop": "12px"}),
                html.H3("Metrics", style=SECTION),
                html.Div(id="metrics-table"),
                dcc.Dropdown(
                    id="metric-dd", clearable=False, style={"width": "260px", "marginTop": "8px"}
                ),
                dcc.Graph(id="fold-fig"),
                html.H3("Learning curve", style=SECTION),
                dcc.RadioItems(
                    [
                        {"label": "Panel per fold", "value": "panels"},
                        {"label": "Overlay folds", "value": "overlay"},
                    ],
                    "panels",
                    id="lc-mode",
                    inline=True,
                ),
                dcc.Graph(id="lc-fig"),
                dcc.Graph(id="gap-fig"),
                html.H3("Feature importance", style=SECTION),
                dcc.Dropdown(id="imp-method", clearable=False, style={"width": "260px"}),
                dcc.Graph(id="imp-fig"),
                html.H3("Predictions and cutoff", style=SECTION),
                html.Div(
                    [
                        dcc.Graph(id="pr-fig", style={"flex": 1}),
                        dcc.Graph(id="rel-fig", style={"flex": 1}),
                    ],
                    style={"display": "flex"},
                ),
                dcc.Slider(
                    0,
                    1,
                    0.01,
                    value=0.5,
                    id="thr-slider",
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                    tooltip={"placement": "bottom"},
                ),
                html.Div(id="thr-text", style={"marginTop": "8px"}),
                dcc.Graph(id="hist-fig"),
                html.H3("Estimated OOF thresholds", style=SECTION),
                html.Div(id="run-threshold-analysis"),
                html.H3("Errors (out-of-fold)", style=SECTION),
                html.Div(id="errors-table"),
                html.H3("Cost", style=SECTION),
                html.Div(id="cost-text"),
                dcc.Graph(id="cost-fig"),
            ],
            style={
                "maxWidth": "1200px",
                "margin": "0 auto",
                "padding": "16px",
                "fontFamily": "sans-serif",
            },
        ),
        engine,
        default,
    )


def register_callbacks(app: Dash, engine: Engine) -> None:

    @app.callback(
        Output("run-dd", "options"), Output("run-dd", "value"), Input("experiment-dd", "value")
    )
    def _runs(experiment):
        return run_choices(engine, experiment)

    @app.callback(Output("run-header", "children"), Input("run-dd", "value"))
    def _header(run_id):
        return header_view(engine, run_id)

    @app.callback(
        Output("metric-dd", "options"),
        Output("metric-dd", "value"),
        Output("metrics-table", "children"),
        Input("run-dd", "value"),
    )
    def _metrics(run_id):
        return metrics_view(engine, run_id)

    @app.callback(
        Output("fold-fig", "figure"), Input("run-dd", "value"), Input("metric-dd", "value")
    )
    def _fold(run_id, metric):
        if not run_id or not metric:
            return rf.empty_fig("No metric selected")
        return rf.fold_metric_fig(rd.load_metrics(engine, run_id), metric)

    @app.callback(
        Output("lc-fig", "figure"),
        Output("gap-fig", "figure"),
        Input("run-dd", "value"),
        Input("lc-mode", "value"),
    )
    def _curves(run_id, mode):
        return curves_view(engine, run_id, mode == "overlay")

    @app.callback(
        Output("imp-method", "options"), Output("imp-method", "value"), Input("run-dd", "value")
    )
    def _methods(run_id):
        return importance_methods(engine, run_id)

    @app.callback(
        Output("imp-fig", "figure"), Input("run-dd", "value"), Input("imp-method", "value")
    )
    def _importance(run_id, method):
        return importance_view(engine, run_id, method)

    @app.callback(Output("pr-fig", "figure"), Output("rel-fig", "figure"), Input("run-dd", "value"))
    def _scoring(run_id):
        return scoring_figs(engine, run_id)

    @app.callback(
        Output("hist-fig", "figure"),
        Output("thr-text", "children"),
        Input("run-dd", "value"),
        Input("thr-slider", "value"),
    )
    def _threshold(run_id, threshold):
        return threshold_view(engine, run_id, threshold)

    @app.callback(Output("errors-table", "children"), Input("run-dd", "value"))
    def _errors(run_id):
        return errors_view(engine, run_id)

    @app.callback(
        Output("cost-fig", "figure"), Output("cost-text", "children"), Input("run-dd", "value")
    )
    def _cost(run_id):
        return cost_view(engine, run_id)

    @app.callback(
        Output("run-threshold-analysis", "children"),
        Input("run-dd", "value"),
        Input("thr-slider", "value"),
    )
    def _threshold_estimates(run_id, threshold):
        if run_id and _reg_data.is_regression_run(engine, run_id):
            return html.P("Classification thresholds do not apply to regression")
        return threshold_panels.render_single(engine, run_id, threshold)

    _reg_page.register_callbacks(app, engine)
