"""Regression Compare, ranking and frozen-evaluation views; no execution callbacks."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ml_template.dashboard import regression_figs as figures
from ml_template.dashboard import regression_multi_data as data
from ml_template.dashboard.pages.regression_run_page import singapore, table


def mode_id(kind):
    return "part2-" + kind + "-task"


def wrap(root, engine, kind):
    original = root.children
    panel = compare_layout() if kind == "compare" else ranking_layout(kind)
    root.children = [
        html.Label("Task"),
        dcc.RadioItems(
            id=mode_id(kind),
            options=[
                {"label": "Binary (existing views)", "value": "binary"},
                {"label": "Regression", "value": "regression"},
            ],
            value="binary",
            inline=True,
        ),
        html.Div(original, id="part2-" + kind + "-legacy"),
        html.Div(panel, id="part2-" + kind + "-reg", style={"display": "none"}),
    ]
    return root


def compare_layout():
    p = "reg2-compare-"
    return html.Div(
        [
            html.H3("Regression comparison — development evidence only"),
            html.Button("Refresh registry/artifacts", id=p + "refresh", n_clicks=0),
            html.Div(id=p + "note"),
            html.Label("Dataset | Silver | split | primary metric | target unit"),
            dcc.Dropdown(id=p + "cohort", options=[], value=None),
            html.Label("Choose at least two runs explicitly"),
            dcc.Dropdown(id=p + "runs", options=[], value=[], multi=True),
            html.Label("Baseline"),
            dcc.Dropdown(id=p + "baseline", options=[], value=None),
            html.Label("Comparison metric — direction is automatic"),
            dcc.Dropdown(
                id=p + "metric", options=["rmse", "mae", "r2"], value="rmse", clearable=False
            ),
            html.Label(
                "Optional saved analyses: at most one per selected run; no automatic latest"
            ),
            dcc.Dropdown(id=p + "analyses", options=[], value=[], multi=True),
            html.Div(id=p + "body"),
        ]
    )


def ranking_layout(kind):
    p = "reg2-" + kind + "-"
    return html.Div(
        [
            html.H3("Regression recipe/seed ranking"),
            html.P(
                "RMSE/MAE minimise; R² maximises. Only one compatible cohort at a time. No holdout metrics enter ranking."
            ),
            html.Button("Refresh registry/artifacts", id=p + "refresh", n_clicks=0),
            html.Div(id=p + "note"),
            html.Label("Sweep launch" if kind == "sweeps" else "Launch scope"),
            dcc.Dropdown(id=p + "launch", options=[], value=None),
            html.Label("Dataset | Silver | split | primary metric | target unit"),
            dcc.Dropdown(id=p + "cohort", options=[], value=None),
            html.Div(id=p + "body"),
        ]
    )


def _mode_callbacks(app, kind):
    @app.callback(
        Output("part2-" + kind + "-legacy", "style"),
        Output("part2-" + kind + "-reg", "style"),
        Input(mode_id(kind), "value"),
    )
    def visible(mode):
        return ({"display": "none"}, {}) if mode == "regression" else ({}, {"display": "none"})


def compare_view(engine, cohort, run_ids, baseline, metric, selections):
    if not cohort or not run_ids or len(run_ids) < 2:
        return html.P("Select one cohort and at least two compatible runs")
    try:
        records, _ = data.inventory(engine)
        eligible = {r["run_id"] for r in data.select_cohort(records, cohort)}
        if not set(run_ids) <= eligible:
            raise ValueError("Selected runs do not belong to the chosen cohort")
        analyses = {}
        for item in selections or []:
            rid, selected = json.loads(item)
            if rid in analyses:
                raise ValueError("Select at most one analysis per run")
            analyses[rid] = selected
        bundle = data.comparison(engine, run_ids, metric, baseline, analyses)
        children = [
            html.H4("Recorded pooled OOF metrics"),
            table(bundle["point"]),
            html.P(
                "Positive improvement_vs_baseline is favourable for either metric direction. Deltas are descriptive, not significance tests. Missing metrics stay unavailable."
            ),
            html.Ul([html.Li(w) for w in bundle["warnings"]]),
        ]
        fold_fig = go.Figure()
        if not bundle["folds"].empty:
            for rid, frame in bundle["folds"].groupby("run_id"):
                frame = frame.sort_values("fold")
                fold_fig.add_trace(
                    go.Scatter(x=frame.fold, y=frame.value, mode="lines+markers", name=rid)
                )
        fold_fig.update_layout(
            title="Stored validation-fold scores",
            xaxis_title="fold",
            yaxis_title=metric
            + (" (dimensionless)" if metric == "r2" else " (" + bundle["unit"] + ")"),
        )
        children.append(dcc.Graph(figure=fold_fig))
        reference = bundle["predictions"][bundle["baseline"]]
        fig, deltas = go.Figure(), go.Figure()
        for rid, frame in bundle["predictions"].items():
            if rid == bundle["baseline"]:
                continue
            positions = (
                np.arange(len(frame))
                if len(frame) <= 5000
                else np.sort(np.random.default_rng(0).choice(len(frame), 5000, replace=False))
            )
            fig.add_trace(
                go.Scattergl(
                    x=reference.y_pred_oof.iloc[positions],
                    y=frame.y_pred_oof.iloc[positions],
                    mode="markers",
                    name=rid,
                    marker={"size": 5, "opacity": 0.5},
                )
            )
            deltas.add_trace(
                go.Histogram(
                    x=(frame.y_pred_oof - reference.y_pred_oof).iloc[positions],
                    name=rid,
                    opacity=0.5,
                )
            )
        fig.update_layout(
            title="Aligned OOF prediction agreement — not an accuracy measure",
            xaxis_title="baseline prediction (" + bundle["unit"] + ")",
            yaxis_title="other prediction (" + bundle["unit"] + ")",
        )
        deltas.update_layout(
            title="Prediction minus baseline; up to 5000 displayed rows per run", barmode="overlay"
        )
        children.extend([dcc.Graph(figure=fig), dcc.Graph(figure=deltas)])
        for rid, frame in bundle["residuals"].items():
            children.append(html.H4("Explicitly selected OOF analysis: " + rid))
            children.extend(
                dcc.Graph(figure=figure)
                for figure in figures.residual_figures(frame.reset_index(), bundle["unit"])
            )
        if bundle["all_analyses_selected"]:
            base_frame = bundle["residuals"][bundle["baseline"]]
            paired = base_frame[["fold"]].copy()
            for rid, frame in bundle["residuals"].items():
                paired[rid + " absolute_error_delta"] = (
                    frame.absolute_error - base_frame.absolute_error
                )
            children.extend(
                [
                    html.H4("Paired absolute-error differences on identical saved OOF rows"),
                    table(paired.reset_index()),
                    html.P(
                        "Negative absolute-error delta is favourable. No row intersection, fitted combiner, confidence interval or new evaluation is computed."
                    ),
                ]
            )
        else:
            children.append(
                html.P(
                    "Select one saved analysis for EVERY run to enable paired residual/error comparison. No missing run is silently dropped."
                )
            )
        return html.Div(children)
    except data.base.ERRORS as exc:
        return html.Pre("Regression comparison unavailable: " + str(exc))


def register_compare(app, engine):
    kind, p = "compare", "reg2-compare-"
    _mode_callbacks(app, kind)

    @app.callback(
        Output(p + "cohort", "options"),
        Output(p + "cohort", "value"),
        Output(p + "note", "children"),
        Input(mode_id(kind), "value"),
        Input(p + "refresh", "n_clicks"),
    )
    def cohorts(mode, clicks):
        if mode != "regression":
            return [], None, ""
        records, warnings = data.inventory(engine)
        missing = sum(r.get("verified") and not r.get("target_unit") for r in records)
        return (
            data.cohort_options(records),
            None,
            html.Div(
                [
                    html.P(
                        f"{missing} verified runs have unknown/conflicting units; record units using the diagnostic CLI."
                    ),
                    html.Ul([html.Li(w) for w in warnings]),
                ]
            ),
        )

    @app.callback(
        Output(p + "runs", "options"),
        Output(p + "runs", "value"),
        Input(p + "cohort", "value"),
        Input(mode_id(kind), "value"),
    )
    def runs(cohort, mode):
        if mode != "regression" or not cohort:
            return [], []
        records, _ = data.inventory(engine)
        chosen = data.select_cohort(records, cohort)
        return [
            {"label": str(r.get("experiment_name")) + " | " + r["run_id"], "value": r["run_id"]}
            for r in chosen
        ], []

    @app.callback(
        Output(p + "baseline", "options"),
        Output(p + "baseline", "value"),
        Output(p + "metric", "value"),
        Output(p + "analyses", "options"),
        Output(p + "analyses", "value"),
        Input(p + "runs", "value"),
        Input(p + "cohort", "value"),
    )
    def choices(run_ids, cohort):
        if not run_ids:
            return [], None, "rmse", [], []
        records, _ = data.inventory(engine)
        chosen = {r["run_id"]: r for r in data.select_cohort(records, cohort)}
        options = []
        for rid in run_ids:
            if rid not in chosen:
                return [], None, "rmse", [], []
            options.extend(
                {"label": rid + " | " + item["label"], "value": json.dumps([rid, item["value"]])}
                for item in chosen[rid]["_analyses"]
            )
        return run_ids, run_ids[0], chosen[run_ids[0]]["primary_metric"], options, []

    @app.callback(
        Output(p + "body", "children"),
        Input(mode_id(kind), "value"),
        Input(p + "cohort", "value"),
        Input(p + "runs", "value"),
        Input(p + "baseline", "value"),
        Input(p + "metric", "value"),
        Input(p + "analyses", "value"),
    )
    def body(mode, cohort, ids, baseline, metric, analyses):
        return (
            compare_view(engine, cohort, ids, baseline, metric, analyses)
            if mode == "regression"
            else html.Div()
        )


def ranking_view(engine, kind, launch, cohort):
    if not cohort:
        return html.P("Select a compatible cohort; unknown units are not ranked")
    try:
        records, warnings = data.inventory(engine)
        scoped = (
            records
            if launch in (None, "all")
            else [r for r in records if r.get("experiment_run_id") == launch]
        )
        selected = data.select_cohort(scoped, cohort)
        board = data.rank_recipes(selected)
        columns = [
            "run_id",
            "experiment_name",
            "experiment_run_id",
            "status",
            "seed",
            "primary_metric",
            "primary_value",
            "recipe_id",
            "target_unit",
            "code_verified",
            "error_summary",
        ]
        detail = pd.DataFrame([{key: r.get(key) for key in columns} for r in selected])
        detail["started_sgt"] = [singapore(r.get("started_at")) for r in selected]
        snapshots = []
        for row in selected:
            snapshot = {"run_id": row["run_id"]}
            snapshot.update(row["_ctx"]["config"]["model"].get("params", {}))
            snapshots.append(snapshot)
        children = [
            html.H4("Recipe ranking — metric direction is automatic"),
            table(board),
            html.P(
                "Latest successful finite-scored run per model seed and recipe. Means average run-level CV scores, not averaged predictions. Std is seed spread, not a confidence interval. Dirty/unknown code or missing environment records remain separate by run, not pooled."
            ),
            html.H4("Verified cohort runs"),
            table(detail),
            html.H4("Recorded model-parameter snapshots"),
            table(pd.DataFrame(snapshots)),
            html.P(
                "Regression fixed-context surfaces and marginal top-k parameter panels are not generated here. Model settings and split seeds remain in recipe identity; this is not an isolated causal parameter effect."
            ),
        ]
        if kind == "sweeps" and launch:
            children.insert(0, table(pd.DataFrame([data.sweep_progress(engine, launch)])))
        incomplete = [r for r in scoped if not r.get("eligible")]
        if incomplete:
            children.extend(
                [
                    html.H4(
                        "Incomplete/unranked registry records (not included in cohort statistics)"
                    ),
                    table(
                        pd.DataFrame(
                            [
                                {
                                    k: r.get(k)
                                    for k in ["run_id", "status", "reason", "error_summary"]
                                }
                                for r in incomplete
                            ]
                        )
                    ),
                ]
            )
        children.append(html.Ul([html.Li(w) for w in warnings]))
        return html.Div(children)
    except data.base.ERRORS as exc:
        return html.Pre("Regression ranking unavailable: " + str(exc))


def register_ranking(app, engine, kind):
    p = "reg2-" + kind + "-"
    _mode_callbacks(app, kind)

    @app.callback(
        Output(p + "launch", "options"),
        Output(p + "launch", "value"),
        Output(p + "note", "children"),
        Input(mode_id(kind), "value"),
        Input(p + "refresh", "n_clicks"),
    )
    def launches(mode, clicks):
        if mode != "regression":
            return [], None, ""
        records, warnings = data.inventory(engine)
        if kind == "sweeps":
            related = {r.get("experiment_run_id") for r in records}
            options = [
                {
                    "label": r["experiment_name"] + " | " + r["experiment_run_id"],
                    "value": r["experiment_run_id"],
                }
                for r in data.launches(engine)
                if r["experiment_run_id"] in related
            ]
            selected = None
        else:
            options, selected = (
                [{"label": "All launches (cohort still required)", "value": "all"}],
                "all",
            )
        return options, selected, html.Ul([html.Li(w) for w in warnings])

    @app.callback(
        Output(p + "cohort", "options"),
        Output(p + "cohort", "value"),
        Input(mode_id(kind), "value"),
        Input(p + "launch", "value"),
        Input(p + "refresh", "n_clicks"),
    )
    def cohorts(mode, launch, clicks):
        if mode != "regression" or (kind == "sweeps" and not launch):
            return [], None
        records, _ = data.inventory(engine)
        scoped = (
            records
            if launch in (None, "all")
            else [r for r in records if r.get("experiment_run_id") == launch]
        )
        return data.cohort_options(scoped), None

    @app.callback(
        Output(p + "body", "children"),
        Input(mode_id(kind), "value"),
        Input(p + "launch", "value"),
        Input(p + "cohort", "value"),
    )
    def body(mode, launch, cohort):
        return ranking_view(engine, kind, launch, cohort) if mode == "regression" else html.Div()


def evaluation_controls():
    return html.Div(
        [
            dcc.Checklist(
                id="reg2-eval-show",
                options=[
                    {
                        "label": "Explicitly load saved labelled holdout prediction plots",
                        "value": "show",
                    }
                ],
                value=[],
            ),
            html.Div(id="reg2-eval-predictions"),
        ],
        id="reg2-eval-controls",
        style={"display": "none"},
    )


def evaluation_view(engine, evaluation_id):
    loaded = data.load_regression_evaluation(engine, evaluation_id, predictions=False)
    if loaded["state"] != "verified":
        return html.Pre("Regression evaluation unavailable: " + loaded["message"])
    document = loaded["data"]
    acceptance = document["acceptance"]
    rows = acceptance.get("results", acceptance.get("criteria", []))
    return html.Div(
        [
            html.H3("Frozen regression holdout evaluation — not development CV"),
            html.P(loaded["message"]),
            html.P("Recorded access: " + singapore(loaded["record"].get("accessed_at"))),
            html.P(
                "Frozen metrics only; no evaluation is run here. Prior exposure is recorded history, not a claim of untouched data. "
                + (
                    "Frozen target unit: "
                    + str(document["summary"]["target_unit"])
                    + ". R2 is dimensionless; prediction intervals are not metric confidence intervals."
                    if document["summary"].get("model_type") == "nn"
                    else "Units were not frozen by this evaluator: shown in native target units."
                )
            ),
            table(pd.DataFrame([document["summary"]])),
            html.H4("Saved holdout metrics"),
            table(pd.DataFrame(document["metrics"])),
            html.H4("Acceptance — not automatic deployment approval"),
            html.P(str(acceptance["status"])),
            table(pd.DataFrame(rows)),
            html.H4("Saved residual summary"),
            html.Pre(json.dumps(document["residual_summary"], indent=2)),
            html.Ul([html.Li(str(w)) for w in document.get("warnings", [])]),
            html.Details(
                [
                    html.Summary("Frozen specification"),
                    html.Pre(json.dumps(document["spec"], indent=2)),
                ]
            ),
            html.Details(
                [
                    html.Summary("Recorded provenance"),
                    html.Pre(json.dumps(document["provenance"], indent=2)),
                ]
            ),
        ]
    )


def register_evaluation(app, engine):
    @app.callback(
        Output("reg2-eval-controls", "style"),
        Output("reg2-eval-predictions", "children"),
        Input("final-eval-id", "value"),
        Input("reg2-eval-show", "value"),
    )
    def saved(evaluation_id, toggle):
        if not evaluation_id:
            return {"display": "none"}, html.Div()
        loaded = data.load_regression_evaluation(
            engine, evaluation_id, predictions=isinstance(toggle, list) and "show" in toggle
        )
        if loaded["state"] != "verified":
            return {"display": "none"}, html.Div()
        if loaded["predictions"] is None:
            return {}, html.P(
                "Saved labelled predictions have not been read. Enable explicitly to display them; source datasets/models are never opened."
            )
        frame = loaded["predictions"].rename(columns={"y_pred": "y_pred_oof"})
        unit = loaded["data"]["summary"].get("target_unit") or "native target units"
        charts = figures.residual_figures(frame, unit)
        for fig in charts:
            if fig.layout.title.text:
                fig.update_layout(title=fig.layout.title.text.replace("OOF", "saved holdout"))
        return {}, html.Div(
            [
                html.P(
                    "Already evaluated, saved holdout predictions — not new evaluation or OOF. Existing metrics above are not recomputed."
                ),
                *[dcc.Graph(figure=fig) for fig in charts],
                table(loaded["predictions"].head(25)),
            ]
        )
