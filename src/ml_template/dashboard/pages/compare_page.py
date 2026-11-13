"""Two-or-more-run comparison page. No model fitting or holdout evaluation."""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard import compare_data as cd
from ml_template.dashboard import compare_figs as cf
from ml_template.dashboard import experiment_data as ed
from ml_template.dashboard import regression_multi_data as _r2
from ml_template.dashboard import regression_multi_views as _r2v
from ml_template.dashboard import threshold_panels
from ml_template.dashboard.inventory_common import grid

P = "compare-"


def layout(engine: Engine):
    runs = _r2.binary_runs(engine)
    runs = runs[runs["status"] == "succeeded"]
    options = ed.cohort_options(runs)
    return _r2v.wrap(
        html.Div(
            [
                html.H2("Compare runs"),
                html.P(
                    "Stored development/CV results only. Select at least two successful runs; F5 refreshes."
                ),
                html.Label("Silver version | stored split | primary metric"),
                dcc.Dropdown(
                    id=P + "cohort",
                    options=options,
                    value=options[0]["value"] if options else None,
                    clearable=False,
                ),
                html.Label("Runs (default: two latest successful runs in the cohort)"),
                dcc.Dropdown(id=P + "runs", options=[], value=[], multi=True),
                html.Label("Baseline"),
                dcc.Dropdown(id=P + "baseline", clearable=False),
                html.Label("Metric for deltas and fold plot"),
                dcc.Dropdown(id=P + "metric", clearable=False),
                dcc.RadioItems(
                    id=P + "direction",
                    options=[
                        {"label": "Higher is better", "value": "higher"},
                        {"label": "Lower is better", "value": "lower"},
                    ],
                    value="higher",
                    inline=True,
                ),
                html.Label("Importance method"),
                dcc.Dropdown(id=P + "method", clearable=False),
                html.Label("Diagnostic curve metric"),
                dcc.Dropdown(id=P + "curve-metric", clearable=False),
                html.Label("Diagnostic curve fold"),
                dcc.Dropdown(id=P + "curve-fold", clearable=False),
                dcc.Checklist(
                    id=P + "diff-only",
                    options=[{"label": "Show only changed parameters", "value": "changed"}],
                    value=["changed"],
                ),
                html.H3("Operating point"),
                dcc.RadioItems(
                    id=P + "mode",
                    options=[
                        {"label": "Common probability cutoff", "value": "threshold"},
                        {"label": "Equal alert budget", "value": "budget"},
                    ],
                    value="budget",
                    inline=True,
                ),
                html.Label("Probability cutoff"),
                dcc.Slider(
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    id=P + "cutoff",
                    marks={i / 10: f"{i / 10:.1f}" for i in range(11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Label("Alert budget (fraction of development rows)"),
                dcc.Slider(
                    0.01,
                    0.3,
                    0.01,
                    value=0.05,
                    id=P + "budget",
                    marks={0.01: "1%", 0.05: "5%", 0.1: "10%", 0.2: "20%", 0.3: "30%"},
                ),
                html.Div(id=P + "body"),
            ],
            style={
                "maxWidth": "1400px",
                "margin": "auto",
                "padding": "16px",
                "fontFamily": "sans-serif",
            },
        ),
        engine,
        "compare",
    )


def build_view(
    bundle: dict,
    baseline: str,
    metric: str,
    higher: bool,
    method: str | None,
    curve_metric: str | None,
    fold: int | None,
    only_changed: bool,
    mode: str,
    cutoff: float,
    budget: float,
):
    headers = bundle["headers"]
    ids = headers["run_id"].tolist()
    metrics, predictions = (bundle["metrics"], bundle["predictions"])
    operating, errors = (pd.DataFrame(), pd.DataFrame())
    if predictions:
        operating, errors = cd.prediction_tables(predictions, baseline, mode, cutoff, budget)
    curves_present = [
        rid
        for rid, frame in bundle["curves"].items()
        if frame is not None
        and (not frame.empty)
        and ((frame["metric"] == curve_metric) & (frame["fold"] == fold)).any()
    ]
    summary_cols = [
        "run_id",
        "experiment_name",
        "seed",
        "feature_set_id",
        "model_type",
        "primary_metric",
        "primary_value",
        "n_features",
        "n_rounds_final",
        "duration_s",
        "started_at",
    ]
    children = [
        html.H3("Run summary"),
        grid(headers.reindex(columns=summary_cols)),
        html.Ul([html.Li(message) for message in bundle["warnings"]]),
        html.H3("Recorded CV metrics"),
        grid(cd.metric_table(metrics, ids)),
        html.P(
            "Fold -1 means pooled OOF values; explicitly named fold_mean/std rows are separate summaries. Pooled score differences are descriptive, not significance tests."
        ),
        dcc.Graph(figure=cf.metric_deltas(metrics, baseline, metric, higher)),
        dcc.Graph(figure=cf.fold_overlay(metrics, metric)),
        html.H3("Prediction quality"),
        dcc.Graph(figure=cf.pr_overlay(predictions)),
        dcc.Graph(figure=cf.reliability_overlay(predictions)),
        html.P(
            "Reliability uses stored OOF probabilities, not newly calibrated scores. Class weighting can change their probability interpretation."
        ),
        html.H3("Estimated OOF thresholds"),
        threshold_panels.render_threshold_panel(
            predictions, current_threshold=cutoff, budget=budget
        ),
        html.H3("Operating-point comparison"),
        grid(operating),
        html.P(
            "Budget mode flags ceil(rate × n) rows per run. Ties use ascending entity-key order. Cutoff/budget choices are exploratory development diagnostics, not validated deployment policies."
        ),
        html.H3("Prediction disagreements and positives missed by every run"),
        grid(errors, 400),
        html.P(
            "This is the full aligned OOF population, not error_sample.csv. The table shows operating-point disagreements or positives missed by all selected runs."
        ),
        html.H3("Input-feature membership"),
        grid(cd.feature_membership(bundle["features"])),
        html.H3("Feature importance"),
        dcc.Graph(figure=cf.importance_heatmap(bundle["importance"], ids, method)),
        html.P(
            "Compare one importance method at a time. Missing features are blank, not zero. Correlated or derived inputs can share permutation-importance credit."
        ),
        html.H3("Diagnostic learning curves"),
        html.P(
            f"Runs with this curve metric/fold: {len(curves_present)} of {len(ids)}. These are separate diagnostic fits, not the original early-stopping histories."
        ),
        dcc.Graph(figure=cf.learning_overlay(bundle["curves"], curve_metric, fold)),
        html.H3("Parameter differences"),
        grid(cd.parameter_diff(bundle["params"], only_changed), 400),
        html.H3("Run cost"),
        dcc.Graph(figure=cf.cost_plot(headers)),
        html.P(
            "Duration includes enabled analyses; it is neither inference latency nor time per boosting round. No holdout metrics or statistical confidence claims are shown."
        ),
    ]
    return html.Div(children)


def register_callbacks(app: Dash, engine: Engine):

    @app.callback(
        Output(P + "runs", "options"), Output(P + "runs", "value"), Input(P + "cohort", "value")
    )
    def _runs(cohort):
        runs = ed.select_cohort(_r2.binary_runs(engine), cohort)
        runs = runs[runs["status"] == "succeeded"].sort_values(
            ["started_at", "run_id"], ascending=False
        )
        options = [
            {"value": r.run_id, "label": f"{r.experiment_name} | seed {r.seed} | {r.run_id}"}
            for r in runs.itertuples()
        ]
        return (options, runs["run_id"].head(2).tolist())

    @app.callback(
        Output(P + "baseline", "options"),
        Output(P + "baseline", "value"),
        Output(P + "metric", "options"),
        Output(P + "metric", "value"),
        Output(P + "method", "options"),
        Output(P + "method", "value"),
        Output(P + "curve-metric", "options"),
        Output(P + "curve-metric", "value"),
        Output(P + "curve-fold", "options"),
        Output(P + "curve-fold", "value"),
        Input(P + "runs", "value"),
        Input(P + "cohort", "value"),
    )
    def _options(run_ids, cohort):
        if not run_ids or len(run_ids) < 2:
            return ([], None, [], None, [], None, [], None, [], None)
        try:
            runs = ed.select_cohort(_r2.binary_runs(engine), cohort)
            selected = cd.selected_runs(runs, run_ids)
            bundle = cd.load_comparison(engine, selected)
        except (ValueError, KeyError, OSError):
            return ([], None, [], None, [], None, [], None, [], None)
        metrics = bundle["metrics"]
        names = sorted(
            metrics.loc[(metrics["split"] == "cv") & (metrics["fold"] == -1), "metric"].unique()
        )
        primary = selected["primary_metric"].iloc[0]
        methods = sorted(bundle["importance"]["method"].unique())
        curves = [
            frame for frame in bundle["curves"].values() if frame is not None and (not frame.empty)
        ]
        curve_metrics = sorted({value for frame in curves for value in frame["metric"]})
        folds = sorted({int(value) for frame in curves for value in frame["fold"]})
        return (
            run_ids,
            run_ids[0],
            names,
            primary if primary in names else names[0] if names else None,
            methods,
            methods[0] if methods else None,
            curve_metrics,
            curve_metrics[0] if curve_metrics else None,
            folds,
            folds[0] if folds else None,
        )

    @app.callback(
        Output(P + "body", "children"),
        Input(P + "runs", "value"),
        Input(P + "cohort", "value"),
        Input(P + "baseline", "value"),
        Input(P + "metric", "value"),
        Input(P + "direction", "value"),
        Input(P + "method", "value"),
        Input(P + "curve-metric", "value"),
        Input(P + "curve-fold", "value"),
        Input(P + "diff-only", "value"),
        Input(P + "mode", "value"),
        Input(P + "cutoff", "value"),
        Input(P + "budget", "value"),
    )
    def _body(
        run_ids,
        cohort,
        baseline,
        metric,
        direction,
        method,
        curve_metric,
        fold,
        changes,
        mode,
        cutoff,
        budget,
    ):
        if not run_ids or len(run_ids) < 2:
            return html.P("Select at least two successful runs")
        if baseline not in run_ids:
            return html.P("Choose a baseline from the selected runs")
        try:
            runs = ed.select_cohort(_r2.binary_runs(engine), cohort)
            selected = cd.selected_runs(runs, run_ids)
            bundle = cd.load_comparison(engine, selected)
            return build_view(
                bundle,
                baseline,
                metric,
                direction == "higher",
                method,
                curve_metric,
                fold,
                "changed" in (changes or []),
                mode,
                cutoff,
                budget,
            )
        except (ValueError, KeyError, OSError) as exc:
            return html.Pre(f"Comparison unavailable: {exc}")

    _r2v.register_compare(app, engine)
