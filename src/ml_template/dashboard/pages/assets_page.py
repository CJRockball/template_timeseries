"""Dataset, feature and model assets page."""

from __future__ import annotations

import json

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard import asset_data as ad
from ml_template.dashboard import experiment_data as ed
from ml_template.dashboard.inventory_common import grid
from ml_template.paths import ROOT

P = "assets-"


def layout(engine: Engine):
    assets = ad.load_assets(engine, ROOT)
    runs = ed.load_experiment_runs(engine)
    cohorts = ed.cohort_options(runs)
    datasets = assets["silver_datasets"]
    sets = assets["feature_sets"]
    return html.Div(
        [
            html.H2("Assets"),
            html.P(
                f"Registered Silver builds: {len(datasets)} | feature blocks: "
                f"{len(assets['feature_blocks'])} | feature sets: {len(sets)} | model runs: {len(runs)}"
            ),
            html.P(
                "Read-only registry inventory. File presence is checked, not file hashes. F5 refreshes."
            ),
            html.H3("Datasets — registered Silver builds"),
            grid(datasets),
            dcc.Dropdown(
                id=P + "dataset",
                options=datasets["silver_dataset_version_id"].tolist(),
                value=datasets["silver_dataset_version_id"].iloc[-1] if len(datasets) else None,
                placeholder="Inspect a dataset manifest",
                clearable=False,
            ),
            html.Pre(id=P + "manifest", style={"maxHeight": "300px", "overflow": "auto"}),
            html.H3("Feature blocks"),
            grid(assets["feature_blocks"]),
            html.H3("Feature-set lineage"),
            grid(ad.lineage_table(sets)),
            dcc.Dropdown(
                id=P + "set",
                options=sets["feature_set_id"].tolist(),
                value=sets["feature_set_id"].iloc[-1] if len(sets) else None,
                placeholder="Inspect final ordered input columns",
                clearable=False,
            ),
            html.Div(id=P + "columns"),
            html.H3("Model leaderboard — CV"),
            html.Label("Silver version | stored split | primary metric"),
            dcc.Dropdown(
                id=P + "cohort",
                options=cohorts,
                value=cohorts[0]["value"] if cohorts else None,
                clearable=False,
            ),
            dcc.RadioItems(
                id=P + "direction",
                options=[
                    {"label": "Higher score is better", "value": "higher"},
                    {"label": "Lower score is better", "value": "lower"},
                ],
                value="higher",
                inline=True,
            ),
            html.P(
                "Top 10 individual successful fits. For seed stability use the Experiments tab."
            ),
            html.Div(id=P + "top"),
            html.Label("Choose additional model runs from this cohort"),
            dcc.Dropdown(id=P + "models", options=[], value=[], multi=True),
            html.Div(id=P + "chosen"),
            dcc.Graph(id=P + "features-score"),
        ],
        style={
            "maxWidth": "1400px",
            "margin": "auto",
            "padding": "16px",
            "fontFamily": "sans-serif",
        },
    )


def register_callbacks(app: Dash, engine: Engine):
    @app.callback(Output(P + "manifest", "children"), Input(P + "dataset", "value"))
    def _manifest(dataset):
        table = ad.load_assets(engine, ROOT)["silver_datasets"]
        row = table[table["silver_dataset_version_id"] == dataset]
        if row.empty:
            return "No dataset selected or registered"
        try:
            return json.dumps(ad.dataset_details(row.iloc[0].to_dict(), ROOT), indent=2)
        except (OSError, ValueError) as exc:
            return f"Cannot read manifest: {exc}"

    @app.callback(Output(P + "columns", "children"), Input(P + "set", "value"))
    def _columns(set_id):
        tables = ad.load_assets(engine, ROOT)
        return grid(ad.feature_columns(tables["feature_sets"], tables["feature_blocks"], set_id))

    @app.callback(
        Output(P + "top", "children"),
        Output(P + "models", "options"),
        Output(P + "models", "value"),
        Output(P + "features-score", "figure"),
        Input(P + "cohort", "value"),
        Input(P + "direction", "value"),
    )
    def _models(cohort, direction):
        runs = ed.select_cohort(ed.load_experiment_runs(engine), cohort)
        ranked = ad.ranked_models(runs, higher=direction == "higher")
        fig = go.Figure(
            go.Scatter(
                x=ranked["n_features"],
                y=ranked["primary_value"],
                mode="markers",
                text=ranked["run_id"],
                hovertemplate="%{text}<br>features %{x}<br>score %{y:.5f}<extra></extra>",
            )
        )
        metric = runs["primary_metric"].iloc[0] if len(runs) else "CV score"
        fig.update_layout(
            template="plotly_white",
            height=340,
            xaxis_title="Raw model input columns",
            yaxis_title=metric,
        )
        options = [
            {"value": row.run_id, "label": f"#{row.rank} | {row.experiment_name} | {row.run_id}"}
            for row in ranked.itertuples()
        ]
        return grid(ranked.head(10)), options, [], fig

    @app.callback(
        Output(P + "chosen", "children"), Input(P + "models", "value"), Input(P + "cohort", "value")
    )
    def _chosen(run_ids, cohort):
        runs = ed.select_cohort(ed.load_experiment_runs(engine), cohort)
        return grid(runs[runs["run_id"].isin(run_ids or [])])
