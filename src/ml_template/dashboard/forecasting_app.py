"""Forecasting-only saved-evidence launcher; no legacy engine/page construction."""

import argparse
from pathlib import Path

from dash import Dash, dcc, html

from ml_template.dashboard.pages import forecasting_page
from ml_template.dashboard.report_callbacks import export_toolbar, register_exports


def create_app(database):
    database = Path(database).resolve(strict=True)
    if not database.is_file():
        raise FileNotFoundError("Existing forecasting project SQLite file required")
    app = Dash(__name__, title="Forecasting — Development")

    def layout():
        return html.Div(
            [
                export_toolbar(),
                dcc.Tabs(
                    id="dashboard-tab",
                    value="forecasting",
                    children=[dcc.Tab(label="Forecasting — Development", value="forecasting")],
                ),
                html.Div(
                    forecasting_page.layout(database=str(database)), id="forecasting-container"
                ),
            ]
        )

    app.layout = layout
    forecasting_page.register_callbacks(app, database=str(database))
    register_exports(app)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    create_app(args.database).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
