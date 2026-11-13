"""Named experiments, comparable seed summaries and individual run records."""

from __future__ import annotations

import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard import experiment_data as ed
from ml_template.dashboard import regression_multi_data as _r2
from ml_template.dashboard import regression_multi_views as _r2v

PREFIX = "experiments-"


def _grid(table: pd.DataFrame, height: int = 320) -> dag.AgGrid:
    clean = table.round(5).astype(object).where(table.notna(), None)
    return dag.AgGrid(
        rowData=clean.to_dict("records"),
        columnDefs=[{"field": str(c)} for c in clean.columns],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={"pagination": True, "paginationPageSize": 20},
        style={"height": f"{height}px"},
    )


def seed_figure(rows: pd.DataFrame, ranking: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for row in ranking.itertuples():
        part = rows[
            (rows["experiment_name"] == row.experiment_name)
            & (rows["recipe_id"] == row.recipe_id)
            & rows["seed"].notna()
        ]
        fig.add_trace(
            go.Box(
                name=f"{row.experiment_name} / {row.recipe_id[:8]}",
                y=part["primary_value"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                customdata=part[["seed", "run_id"]].to_numpy(),
                hovertemplate="score %{y:.5f}<br>seed %{customdata[0]}<br>%{customdata[1]}<extra>%{fullData.name}</extra>",
            )
        )
    if ranking.empty:
        fig.add_annotation(text="No scored successful runs with recorded seeds", showarrow=False)
    fig.update_layout(
        template="plotly_white",
        height=430,
        showlegend=False,
        yaxis_title="CV primary score",
        margin={"b": 120},
    )
    return fig


def layout(engine: Engine) -> html.Div:
    runs = _r2.binary_runs(engine)
    cohorts = ed.cohort_options(runs)
    names = sorted(runs["experiment_name"].unique())
    latest = None
    if not runs.empty:
        newest = runs.sort_values(["started_at", "run_id"], ascending=False).iloc[0]
        for option in cohorts:
            if not ed.select_cohort(runs.loc[[newest.name]], option["value"]).empty:
                latest = option["value"]
                break
    return _r2v.wrap(
        html.Div(
            [
                html.H2("Named experiments"),
                html.P(
                    "Refresh the browser with F5 after training. All scores below are development/CV scores."
                ),
                html.H3("Inventory — all datasets and statuses"),
                _grid(ed.experiment_inventory(runs), 250),
                html.H3("Seed-sweep leaderboard"),
                html.Label("Comparison cohort: Silver version | stored split | primary metric"),
                dcc.Dropdown(
                    id=PREFIX + "cohort",
                    options=cohorts,
                    value=latest or (cohorts[0]["value"] if cohorts else None),
                    clearable=False,
                ),
                html.Label(
                    "Named experiments (empty = all; table shows every recipe, chart shows top 10)"
                ),
                dcc.Dropdown(id=PREFIX + "names", options=names, value=[], multi=True),
                dcc.RadioItems(
                    id=PREFIX + "direction",
                    options=[
                        {"label": "Higher score is better", "value": "higher"},
                        {"label": "Lower score is better", "value": "lower"},
                    ],
                    value="higher",
                    inline=True,
                ),
                html.P(id=PREFIX + "note"),
                html.Div(id=PREFIX + "leaderboard"),
                dcc.Graph(id=PREFIX + "seeds"),
                html.H3("Runs for the selected experiments and cohort"),
                html.P(
                    "Includes running and failed runs. Use the run ID on the Single run tab to inspect a completed run."
                ),
                html.Div(id=PREFIX + "runs"),
                html.P(
                    "Spread is sample standard deviation across model seeds, not independent-data uncertainty. Unknown code changes in dirty working trees cannot be verified from these records."
                ),
            ],
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "padding": "16px",
                "fontFamily": "sans-serif",
            },
        ),
        engine,
        "experiments",
    )


def build_view(engine: Engine, cohort: str | None, names: list[str] | None, direction: str):
    runs = ed.select_cohort(_r2.binary_runs(engine), cohort)
    if names:
        runs = runs[runs["experiment_name"].isin(names)]
    board = ed.seed_leaderboard(runs, higher_is_better=direction != "lower")
    selected = ed.seed_rows(runs)
    omitted = int(selected["seed"].isna().sum())
    note = f"{len(board)} experiment/recipe groups. Ranking uses one latest successful run per recorded model seed; std is blank for one seed. {omitted} unseeded runs excluded from ranking. Different model settings remain separate rows."
    cols = [
        "experiment_name",
        "experiment_run_id",
        "run_id",
        "status",
        "seed",
        "recipe_id",
        "primary_value",
        "duration_s",
        "n_features",
        "n_rounds_final",
        "started_at",
        "error_summary",
    ]
    detail = runs.sort_values(["started_at", "run_id"], ascending=False)[cols].copy()
    detail["started_at"] = (
        pd.to_datetime(detail["started_at"], utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    detail = detail.rename(columns={"started_at": "started_sgt"})
    return (_grid(board), seed_figure(selected, board.head(10)), _grid(detail, 400), note)


def register_callbacks(app: Dash, engine: Engine) -> None:

    @app.callback(
        Output(PREFIX + "leaderboard", "children"),
        Output(PREFIX + "seeds", "figure"),
        Output(PREFIX + "runs", "children"),
        Output(PREFIX + "note", "children"),
        Input(PREFIX + "cohort", "value"),
        Input(PREFIX + "names", "value"),
        Input(PREFIX + "direction", "value"),
    )
    def _view(cohort, names, direction):
        return build_view(engine, cohort, names, direction)

    _r2v.register_ranking(app, engine, "experiments")
