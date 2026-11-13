"""Seven-tab read-only dashboard, preserving Single run as the default."""

from __future__ import annotations

import argparse

from dash import Dash, Input, Output, dcc, html

from ml_template.dashboard.pages import (
    assets_page,
    calibration_page,
    compare_page,
    evaluation_page,
    experiment_page,
    run_page,
    sweeps_page,
)
from ml_template.dashboard.report_callbacks import export_toolbar, register_exports


def create_app(engine=None):
    if engine is None:
        from ml_template.db.connection import get_engine

        engine = get_engine()
    app = Dash(__name__, title="Model experiments")
    pages = [
        ("assets", "Assets", "assets-container", assets_page),
        ("sweeps", "Sweeps", "sweeps-container", sweeps_page),
        ("experiments", "Experiments", "experiments-container", experiment_page),
        ("run", "Single run", "single-run-container", run_page),
        ("compare", "Compare", "compare-container", compare_page),
        ("calibration", "Calibration", "calibration-container", calibration_page),
        ("evaluation", "Final evaluation", "evaluation-container", evaluation_page),
    ]

    def layout():
        return html.Div(
            [
                export_toolbar(),
                dcc.Tabs(
                    id="dashboard-tab",
                    value="run",
                    children=[
                        dcc.Tab(label=label, value=value)
                        for value, label, _container, _page in pages
                    ],
                ),
                *[
                    html.Div(
                        page.layout(engine),
                        id=container,
                        style={} if value == "run" else {"display": "none"},
                    )
                    for value, _label, container, page in pages
                ],
            ]
        )

    app.layout = layout
    for _value, _label, _container, page in pages:
        page.register_callbacks(app, engine)
    register_exports(app)

    @app.callback(
        *[Output(container, "style") for _value, _label, container, _page in pages],
        Input("dashboard-tab", "value"),
    )
    def tab(value):
        return tuple(
            {} if value == name else {"display": "none"}
            for name, _label, _container, _page in pages
        )

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    create_app().run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
