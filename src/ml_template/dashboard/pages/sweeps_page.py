"""Launch-level sweep progress, fixed-context diagnostics and marginal summaries."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard import experiment_data as ed
from ml_template.dashboard import regression_multi_data as _r2
from ml_template.dashboard import regression_multi_views as _r2v
from ml_template.dashboard import sweep_data as sd
from ml_template.dashboard.inventory_common import grid
from ml_template.dashboard.sweep_figs import parameter_plot, placeholder
from ml_template.dashboard.sweep_parameter_panels import (
    parameter_panels,
    register_parameter_callbacks,
)

P = "sweeps-"


def _launch_frame(engine: Engine, launch: str | None):
    _, runs, params = _r2.binary_sweep_data(engine)
    selected = runs[runs["experiment_run_id"] == launch]
    return sd.parameter_frame(selected, params)


def layout(engine: Engine):
    launches, runs, _ = _r2.binary_sweep_data(engine)
    inventory = sd.launch_summary(launches, runs)
    options = [
        {
            "label": f"{r.experiment_name} | {r.experiment_run_id} | {r.status}",
            "value": r.experiment_run_id,
        }
        for r in inventory.itertuples()
    ]
    return _r2v.wrap(
        html.Div(
            [
                html.H2("Sweeps"),
                html.P(
                    "One launch at a time. F5 refreshes progress; this page never launches training."
                ),
                grid(inventory),
                dcc.Dropdown(
                    id=P + "launch",
                    options=options,
                    value=options[0]["value"] if options else None,
                    clearable=False,
                    placeholder="No sweep launches registered",
                ),
                html.Label("Comparison cohort: Silver version | stored split | primary metric"),
                dcc.Dropdown(id=P + "cohort", clearable=False),
                dcc.RadioItems(
                    id=P + "direction",
                    options=[
                        {"label": "Higher score is better", "value": "higher"},
                        {"label": "Lower score is better", "value": "lower"},
                    ],
                    value="higher",
                    inline=True,
                ),
                html.H3("Configurations ranked over model seeds"),
                html.Div(id=P + "ranking"),
                html.H3("Parameter diagnostics"),
                html.P(
                    "Choose one or two varied parameters, then fix all remaining settings with the context selector. A seed-only sweep has no parameter axes; use its configuration ranking instead."
                ),
                dcc.Dropdown(id=P + "x", placeholder="Parameter X", clearable=False),
                dcc.Dropdown(id=P + "y", placeholder="Parameter Y (optional)", clearable=True),
                dcc.Dropdown(
                    id=P + "context", placeholder="Fixed remaining settings", clearable=False
                ),
                html.P(id=P + "note"),
                dcc.Graph(id=P + "plot"),
                html.Div(id=P + "cells"),
                *parameter_panels(P),
                html.H3("Run records and failure summaries"),
                html.Div(id=P + "runs"),
                html.P(
                    "Blank std means one seed. Error bars show seed spread, not confidence intervals. Missing heatmap cells are unobserved combinations, not interpolated estimates. Summed run durations include enabled analyses and are not sweep wall-clock duration."
                ),
            ],
            style={
                "maxWidth": "1400px",
                "margin": "auto",
                "padding": "16px",
                "fontFamily": "sans-serif",
            },
        ),
        engine,
        "sweeps",
    )


def register_callbacks(app: Dash, engine: Engine):
    register_parameter_callbacks(app, engine, _launch_frame, P)

    @app.callback(
        Output(P + "cohort", "options"), Output(P + "cohort", "value"), Input(P + "launch", "value")
    )
    def _cohorts(launch):
        options = ed.cohort_options(_launch_frame(engine, launch))
        return (options, options[0]["value"] if options else None)

    @app.callback(
        Output(P + "ranking", "children"),
        Output(P + "runs", "children"),
        Output(P + "x", "options"),
        Output(P + "x", "value"),
        Output(P + "y", "options"),
        Output(P + "y", "value"),
        Input(P + "launch", "value"),
        Input(P + "cohort", "value"),
        Input(P + "direction", "value"),
    )
    def _overview(launch, cohort, direction):
        frame = ed.select_cohort(_launch_frame(engine, launch), cohort)
        axes = sd.varied_parameters(frame)
        options = [{"label": c.removeprefix(sd.PARAM_PREFIX), "value": c} for c in axes]
        board = ed.seed_leaderboard(frame, higher_is_better=direction == "higher")
        return (
            grid(board),
            grid(frame.rename(columns=lambda c: c.removeprefix(sd.PARAM_PREFIX))),
            options,
            axes[0] if axes else None,
            options,
            axes[1] if len(axes) > 1 else None,
        )

    @app.callback(
        Output(P + "context", "options"),
        Output(P + "context", "value"),
        Input(P + "launch", "value"),
        Input(P + "cohort", "value"),
        Input(P + "x", "value"),
        Input(P + "y", "value"),
    )
    def _contexts(launch, cohort, x, y):
        frame = ed.select_cohort(_launch_frame(engine, launch), cohort)
        _, options = sd.context_groups(frame, list(dict.fromkeys(a for a in (x, y) if a)))
        return (options, options[0]["value"] if options else None)

    @app.callback(
        Output(P + "plot", "figure"),
        Output(P + "cells", "children"),
        Output(P + "note", "children"),
        Input(P + "launch", "value"),
        Input(P + "cohort", "value"),
        Input(P + "x", "value"),
        Input(P + "y", "value"),
        Input(P + "context", "value"),
        Input(P + "direction", "value"),
    )
    def _plot(launch, cohort, x, y, context, direction):
        axes = [a for a in (x, y) if a]
        frame = ed.select_cohort(_launch_frame(engine, launch), cohort)
        if not x or len(set(axes)) != len(axes):
            return (placeholder("Choose one or two different varied parameters"), html.Div(), "")
        if any(a not in frame for a in axes):
            return (placeholder("Choose parameters from this launch"), html.Div(), "")
        frame, _ = sd.context_groups(frame, axes)
        part = frame[frame["context_id"] == context]
        summary = sd.cell_summary(part, axes)
        note = f"{len(part)} registered runs in this context; {len(summary)} observed scored cells. Only successful runs with recorded seeds and axis values enter cell means. For repeated model seeds, the latest successful scored run is used."
        return (
            parameter_plot(summary, axes, direction == "higher"),
            grid(summary.rename(columns=lambda c: c.removeprefix(sd.PARAM_PREFIX))),
            note,
        )

    _r2v.register_ranking(app, engine, "sweeps")
