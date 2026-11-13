"""Read-only Dash panels for marginal sweep performance and top-k stability."""

from __future__ import annotations

from dash import Input, Output, dcc, html

from ml_template.dashboard import regression_multi_data as _r2
from ml_template.dashboard.inventory_common import grid
from ml_template.dashboard.sweep_declared_parameters import load_launch_summary_inputs
from ml_template.dashboard.sweep_parameter_summary import parameter_summaries


def parameter_panels(prefix: str) -> list:
    return [
        html.H3("Overall parameter performance"),
        html.P(
            "One row per parameter value. Mean and sample std pool all successful finite-scored runs with that value, allowing other settings to vary. Repeated seeds/reruns count separately; this is not an isolated parameter effect. Blank std means fewer than two observations."
        ),
        html.P(id=prefix + "parameter-summary-note"),
        html.Div(id=prefix + "parameter-overall"),
        html.H3(id=prefix + "parameter-top-title"),
        html.Label("Number of top runs"),
        dcc.Dropdown(
            id=prefix + "parameter-k",
            options=[{"label": str(k), "value": k} for k in (5, 10, 20, 50)],
            value=10,
            clearable=False,
            style={"maxWidth": "160px"},
        ),
        html.P(
            "Mode share uses recorded values only; missing_runs reports excluded cells. All tied modes are shown with their individual overall baseline shares. The single baseline column is blank for ties. Fixed throughout means no alternative was tested. Ranking follows metric direction, then ascending run_id for tied scores."
        ),
        html.Div(id=prefix + "parameter-top"),
    ]


def summary_content(frame, *, k: int = 10, higher_is_better: bool = True):
    overall, stability, meta = parameter_summaries(frame, k=k, higher_is_better=higher_is_better)
    n, selected = (meta["eligible_runs"], meta["selected_runs"])
    note = f"{n} successful finite-scored runs of {meta['registered_runs']} registered; {meta['excluded_runs']} excluded. {meta['parameter_count']} parameter columns. Uses the whole selected sweep launch, independent of diagnostics context, grid pagination and filters."
    if len(meta["model_types"]) > 1:
        note += (
            " Mixed model types: "
            + ", ".join(meta["model_types"])
            + ". Interpret pooled summaries cautiously."
        )
    title = f"Top-{selected} of {n} runs: parameter stability (requested {k})"
    overall_component = (
        grid(overall, height=360)
        if not overall.empty
        else html.P("No successful finite-scored runs with recorded parameter values.")
    )
    if meta["insufficient_runs"]:
        top_component = html.P(
            "Insufficient runs: at least two successful scored runs are required."
        )
    elif stability.empty:
        top_component = html.P("No recorded parameter columns in this cohort.")
    else:
        top_component = grid(stability, height=360)
    return (overall_component, top_component, note, title)


def register_parameter_callbacks(app, engine, launch_frame, prefix: str):

    @app.callback(
        Output(prefix + "parameter-overall", "children"),
        Output(prefix + "parameter-top", "children"),
        Output(prefix + "parameter-summary-note", "children"),
        Output(prefix + "parameter-top-title", "children"),
        Input(prefix + "launch", "value"),
        Input(prefix + "cohort", "value"),
        Input(prefix + "direction", "value"),
        Input(prefix + "parameter-k", "value"),
    )
    def _parameter_summary(launch, cohort, direction, k):
        try:
            frame, document, source = load_launch_summary_inputs(engine, launch)
            if _r2.nonbinary_launch(engine, launch, frame=frame):
                message = "Mixed task/metric launch: use task-specific panels; whole-launch binary marginal summary is blocked"
                return (html.P(message), html.P(message), message, "Task-specific summary required")
            overall, top, note, title = summary_content(
                frame, k=10 if k is None else k, higher_is_better=direction == "higher"
            )
        except (OSError, ValueError) as error:
            message = f"Sweep summary unavailable: {error}"
            return (html.P(message), html.P(message), message, "Sweep parameter stability")
        note += f" Sweep YAML: {source}; {len(document['grid'])} grid keys. Only grid-declared parameters are shown. Values and scores come from recorded runs across the whole experiment_run_id. The current YAML file is not an immutable launch-time snapshot."
        return (overall, top, note, title)
