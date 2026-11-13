"""Read-only calibration page. All evidence comes from saved development artifacts."""

from __future__ import annotations

import json

from dash import Dash, Input, Output, State, dcc, html

from ml_template.dashboard import calibration_data as cd
from ml_template.dashboard import calibration_figs as cf
from ml_template.dashboard.inventory_common import grid
from ml_template.dashboard.threshold_panels import render_threshold_panel

P = "calibration-"
STYLE = {"maxWidth": "1400px", "margin": "auto", "padding": "16px", "fontFamily": "sans-serif"}


def _parent_options(runs):
    return [
        {"label": str(value), "value": str(value)} for value in runs.parent_run_id.drop_duplicates()
    ]


def layout(engine):
    runs = cd.list_calibrations(engine)
    options = _parent_options(runs)
    return html.Div(
        [
            html.H2("Calibration"),
            html.P(
                "Read-only development diagnostics and exploratory threshold estimates. No policy freezing or holdout evaluation."
            ),
            html.Button("Refresh saved calibrations", id=P + "refresh", n_clicks=0),
            html.Label("Parent model run"),
            dcc.Dropdown(
                id=P + "parent",
                options=options,
                value=options[0]["value"] if options else None,
                clearable=False,
            ),
            html.Label("Calibration run"),
            dcc.Dropdown(id=P + "id", options=[], value=None, clearable=False),
            html.Label("Per-fold metric"),
            dcc.Dropdown(
                id=P + "metric",
                options=["brier", "log_loss", "pr_auc", "roc_auc"],
                value="brier",
                clearable=False,
            ),
            html.H3("Threshold diagnostics"),
            html.Label("Score variant (cutoffs are not interchangeable)"),
            dcc.Dropdown(
                id=P + "threshold-variant",
                options=[
                    {"label": "Raw OOF", "value": "raw"},
                    {"label": "Calibrated cross-fit", "value": "calibrated_crossfit"},
                ],
                value="calibrated_crossfit",
                clearable=False,
            ),
            html.Label("Exploratory cutoff for the selected score variant"),
            dcc.Input(id=P + "threshold", type="number", min=0, max=1, step="any", value=0.1),
            html.Label("Alert budget (fraction of development rows)"),
            dcc.Input(id=P + "budget", type="number", min=0.000001, max=1, step=0.01, value=0.05),
            html.Div(id=P + "body"),
        ],
        style=STYLE,
    )


def build_view(
    bundle, metric="brier", score_variant="calibrated_crossfit", threshold=0.1, budget=0.05
):
    comparison = cd.metric_comparison(bundle["metrics"])
    distributions = cd.distribution_table(bundle["predictions"])
    summary = (
        comparison.loc[
            comparison.metric.isin(["n_rows", "n_positives", "base_rate", "mean_probability"])
        ]
        if not comparison.empty
        else comparison
    )
    threshold_frames, threshold_metadata = cd.threshold_inputs(bundle, score_variant)
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Exploratory development diagnostics — not final-test results"),
                    html.P(
                        "Each cross-fit prediction excludes its row's fold from the calibrator fit. "
                        "This is not nested evaluation of base-model training or selection. "
                        "The final calibrator is fitted on all development OOF pairs; its in-sample p_cal_fit values are excluded."
                    ),
                ],
                style={"backgroundColor": "#fff3cd", "padding": "12px", "marginTop": "16px"},
            ),
            html.Ul([html.Li(message) for message in bundle["warnings"]]),
            html.H3("Calibration and parent identity"),
            grid(cd.metadata_table(bundle), 370),
            html.H3("Development population and probability scale"),
            grid(summary, 240),
            html.H3("Raw versus calibrated cross-fit metrics"),
            grid(comparison, 340),
            html.P(
                "Delta means calibrated minus raw. Negative Brier/log-loss deltas are favourable. "
                "These are descriptive comparisons, not significance tests. Brier/log loss measure more than calibration alone."
            ),
            dcc.Graph(figure=cf.loss_fig(comparison)),
            dcc.Graph(figure=cf.reliability_fig(bundle["reliability"])),
            html.P(
                "Saved quantile bins retain tied scores and may be fewer than requested. Each variant uses its own bins. "
                "Whiskers are pointwise Wilson intervals assuming independent rows, not model-level confidence intervals."
            ),
            html.H4("Reliability support counts"),
            grid(bundle["reliability"], 340),
            dcc.Graph(figure=cf.fold_metric_fig(bundle["metrics"], metric)),
            html.P(
                "Fold-specific calibration mappings can change pooled ranking metrics. "
                "Do not interpret such changes as evidence that the single final mapping improves ranking."
            ),
            dcc.Graph(figure=cf.distribution_fig(distributions)),
            dcc.Graph(figure=cf.mapping_fig(bundle["calibrator"])),
            html.H3("Estimated thresholds on the selected development score scale"),
            render_threshold_panel(
                threshold_frames, threshold, budget, metadata=threshold_metadata
            ),
            html.P(
                "To freeze a policy, use the explicit policy_selection CLI after reviewing the objective, "
                "workload, ties and score-scale transfer. This page never writes a policy."
            ),
            html.H3("Saved settings and final parameters"),
            html.Pre(
                json.dumps(
                    {"config": bundle["config"], "calibrator": bundle["calibrator"]}, indent=2
                ),
                style={"whiteSpace": "pre-wrap"},
            ),
            html.H3("Artifact integrity"),
            grid(bundle["inventory"], 300),
            html.P(
                "Hashes check saved calibration artifacts, not a new validation of parent data or model performance. "
                "No training/holdout files are read by this page."
            ),
            html.Details(
                [
                    html.Summary("Recorded provenance and diagnostic notes"),
                    html.Pre(
                        json.dumps(
                            {"provenance": bundle["provenance"], "notes": bundle["notes"]}, indent=2
                        ),
                        style={"whiteSpace": "pre-wrap"},
                    ),
                ]
            ),
        ]
    )


def register_callbacks(app: Dash, engine):
    @app.callback(
        Output(P + "parent", "options"),
        Output(P + "parent", "value"),
        Input(P + "refresh", "n_clicks"),
        State(P + "parent", "value"),
    )
    def _parents(_clicks, selected):
        options = _parent_options(cd.list_calibrations(engine))
        values = [item["value"] for item in options]
        return options, selected if selected in values else (values[0] if values else None)

    @app.callback(
        Output(P + "id", "options"),
        Output(P + "id", "value"),
        Input(P + "parent", "value"),
        Input(P + "refresh", "n_clicks"),
        State(P + "id", "value"),
    )
    def _calibrations(parent, _clicks, selected):
        if not parent:
            return [], None
        runs = cd.list_calibrations(engine, parent)
        options = [
            {
                "value": str(r.calibration_id),
                "label": f"{r.calibration_id} | {r.method} | {r.status}",
            }
            for r in runs.itertuples()
        ]
        values = [item["value"] for item in options]
        successful = runs.loc[runs.status.eq("succeeded"), "calibration_id"].tolist()
        default = successful[0] if successful else (values[0] if values else None)
        return options, selected if selected in values else default

    @app.callback(
        Output(P + "body", "children"),
        Input(P + "id", "value"),
        Input(P + "parent", "value"),
        Input(P + "metric", "value"),
        Input(P + "threshold-variant", "value"),
        Input(P + "threshold", "value"),
        Input(P + "budget", "value"),
        Input(P + "refresh", "n_clicks"),
    )
    def _body(calibration_id, parent, metric, score_variant, threshold, budget, _clicks):
        if not calibration_id:
            return html.P("No calibration selected. Run calibration from the CLI, then refresh.")
        try:
            bundle = cd.load_calibration(engine, calibration_id)
            if bundle["header"]["parent_run_id"] != parent:
                return html.P("Choose a calibration for the selected parent run.")
            if (
                threshold is None
                or not 0 <= threshold <= 1
                or budget is None
                or not 0 < budget <= 1
            ):
                return html.P("Use a cutoff in [0,1] and an alert budget in (0,1].")
            return build_view(
                bundle, metric or "brier", score_variant or "calibrated_crossfit", threshold, budget
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return html.Pre(f"Calibration diagnostics unavailable: {exc}")


# calibrated-policy update v1
