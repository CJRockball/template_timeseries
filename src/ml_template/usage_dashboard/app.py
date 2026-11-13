"""Separate read-only explanation dashboard. No inference, fitting or write callbacks."""

import argparse
import json
import sqlite3
from pathlib import Path

from ml_template.explainability.figures import (
    dependence_figure,
    distribution_figure,
    importance_figure,
    placeholder,
    waterfall_figure,
)
from ml_template.explainability.presentation_data import (
    list_explanations,
    load_explanation,
    selected_entity,
    table_records,
)


def global_payload(database, explanation_id, *, max_rows=100000):
    view = load_explanation(database, explanation_id, max_rows=max_rows)
    feature_options = [{"label": name, "value": name} for name in view.importance.feature]
    entity_options = [
        {"label": str(key), "value": position} for position, key in enumerate(view.rows.index)
    ]
    table = view.rows[
        [
            name
            for name in (
                "p_raw",
                "p_calibrated",
                "p_release",
                "policy_flag",
                "y_pred_raw",
                "y_pred_release",
            )
            if name in view.rows
        ]
    ].copy()
    table.insert(0, "entity_key", view.rows.index.to_numpy())
    columns = [{"name": name, "id": name} for name in table.columns]
    return (
        json.dumps(view.summary, indent=2),
        feature_options,
        feature_options[0]["value"] if feature_options else None,
        entity_options,
        0 if entity_options else None,
        table_records(table),
        columns,
        importance_figure(view),
        distribution_figure(view),
    )


def local_payload(database, explanation_id, position, *, max_rows=100000):
    view = load_explanation(database, explanation_id, max_rows=max_rows)
    key, row, table = selected_entity(view, position)
    scores = {
        name: str(row[name])
        for name in (
            "p_raw",
            "p_calibrated",
            "p_release",
            "policy_flag",
            "y_pred_raw",
            "y_pred_release",
        )
        if name in row.index
    }
    return (
        waterfall_figure(view, position),
        table_records(table),
        f"Entity {key}: " + json.dumps(scores),
    )


def create_app(database, *, max_rows=100000):
    from dash import Dash, Input, Output, dash_table, dcc, html

    database = Path(database).resolve()
    app = Dash(__name__, title="Released model explanations")

    def layout():
        error = ""
        try:
            inventory = list_explanations(database)
        except (ValueError, OSError, sqlite3.Error) as exc:
            inventory = []
            error = str(exc)
        options = [
            {
                "label": f"{r['explanation_id']} | {r['release_id']} | {r['scoring_batch_id']}",
                "value": r["explanation_id"],
            }
            for r in inventory
        ]
        return html.Div(
            [
                html.H1("Released model explanations"),
                html.P(
                    "Separate from the training workbench. Read-only saved artifacts; no model execution."
                ),
                html.P(
                    error
                    or (
                        "Choose a completed explanation. Refresh the browser to discover new executions."
                        if options
                        else "No completed explanation executions are registered."
                    )
                ),
                dcc.Dropdown(
                    id="usage-explanation",
                    options=options,
                    value=options[0]["value"] if options else None,
                    clearable=False,
                ),
                html.Pre(id="usage-summary"),
                html.P(
                    "Classification contributions explain base-model log-odds, not calibrated probabilities. Regression contributions explain continuous predictions in the declared target units. Neither establishes causes. Global figures describe the complete scoring batch. CV inputs that vary across fitted transforms are explicitly marked; their dependence plots are disabled."
                ),
                dcc.Graph(id="usage-importance"),
                dcc.Graph(id="usage-distribution"),
                html.H2("Feature dependence"),
                dcc.Dropdown(id="usage-feature", clearable=False),
                dcc.Graph(id="usage-dependence"),
                html.H2("Saved predictions"),
                dash_table.DataTable(
                    id="usage-rows",
                    data=[],
                    columns=[],
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                ),
                html.H2("Individual prediction"),
                dcc.Dropdown(id="usage-entity", clearable=False),
                html.Pre(id="usage-scores"),
                dcc.Graph(id="usage-waterfall"),
                dash_table.DataTable(
                    id="usage-contributions",
                    data=[],
                    columns=[
                        {"name": name, "id": name} for name in ("feature", "value", "contribution")
                    ],
                    page_size=15,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                ),
            ],
            style={"maxWidth": "1500px", "margin": "2rem auto", "fontFamily": "system-ui"},
        )

    from ml_template.usage_dashboard.monitoring_page import (
        monitoring_layout,
        register_monitoring_callbacks,
    )

    def combined_layout():
        return html.Div(
            [
                dcc.Tabs(
                    id="usage-section",
                    value="explanations",
                    children=[
                        dcc.Tab(label="Explanations", value="explanations", children=layout()),
                        dcc.Tab(
                            label="Monitoring",
                            value="monitoring",
                            children=monitoring_layout(database),
                        ),
                    ],
                )
            ]
        )

    app.layout = combined_layout

    @app.callback(
        Output("usage-summary", "children"),
        Output("usage-feature", "options"),
        Output("usage-feature", "value"),
        Output("usage-entity", "options"),
        Output("usage-entity", "value"),
        Output("usage-rows", "data"),
        Output("usage-rows", "columns"),
        Output("usage-importance", "figure"),
        Output("usage-distribution", "figure"),
        Input("usage-explanation", "value"),
    )
    def select_explanation(identity):
        if not identity:
            return (
                "No explanation selected",
                [],
                None,
                [],
                None,
                [],
                [],
                placeholder("No data"),
                placeholder("No data"),
            )
        try:
            return global_payload(database, identity, max_rows=max_rows)
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            return (
                f"Artifact verification/loading failed: {exc}",
                [],
                None,
                [],
                None,
                [],
                [],
                placeholder("Unavailable"),
                placeholder("Unavailable"),
            )

    @app.callback(
        Output("usage-dependence", "figure"),
        Input("usage-explanation", "value"),
        Input("usage-feature", "value"),
    )
    def select_feature(identity, feature):
        if not identity or feature is None:
            return placeholder("Choose a feature")
        try:
            return dependence_figure(
                load_explanation(database, identity, max_rows=max_rows), feature
            )
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            return placeholder(str(exc))

    @app.callback(
        Output("usage-waterfall", "figure"),
        Output("usage-contributions", "data"),
        Output("usage-scores", "children"),
        Input("usage-explanation", "value"),
        Input("usage-entity", "value"),
    )
    def select_row(identity, position):
        if not identity or position is None:
            return (placeholder("Choose an entity"), [], "")
        try:
            return local_payload(database, identity, position, max_rows=max_rows)
        except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
            return (placeholder(str(exc)), [], str(exc))

    register_monitoring_callbacks(app, database)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.database is None:
        from ml_template.paths import ROOT

        args.database = ROOT / "data/project.db"
    create_app(args.database, max_rows=args.max_rows).run(
        host="127.0.0.1", port=args.port, debug=args.debug
    )


if __name__ == "__main__":
    main()
