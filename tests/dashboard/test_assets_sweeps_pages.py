import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")
pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine  # noqa: E402

from ml_template.dashboard import asset_data as ad  # noqa: E402
from ml_template.dashboard import sweep_data as sd  # noqa: E402
from ml_template.dashboard.app import create_app  # noqa: E402
from ml_template.dashboard.inventory_common import grid, read_registry  # noqa: E402
from ml_template.dashboard.pages import assets_page, sweeps_page  # noqa: E402
from ml_template.db.tracking_tables import (  # noqa: E402
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_params,
)


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new_pages.db'}")
    ensure_tracking_tables(engine)
    with engine.begin() as conn:
        conn.execute(
            experiment_launches.insert(),
            {
                "experiment_run_id": "sweep1",
                "experiment_name": "grid",
                "kind": "sweep",
                "n_runs": 2,
                "status": "succeeded",
                "started_at": "2026-10-16T00:00:00Z",
            },
        )
        for i in (1, 2):
            rid = f"r{i}"
            conn.execute(
                model_runs.insert(),
                {
                    "run_id": rid,
                    "experiment_run_id": "sweep1",
                    "experiment_name": "grid",
                    "config_hash": str(i),
                    "status": "succeeded",
                    "seed": 1,
                    "started_at": f"2026-10-16T0{i}:00:00Z",
                    "silver_dataset_version_id": "s",
                    "split_id": "sp",
                    "model_type": "xgb",
                    "primary_metric": "pr_auc",
                    "primary_value": 0.70 + i / 100,
                    "duration_s": 10.0,
                },
            )
            conn.execute(
                run_params.insert(),
                [
                    {"run_id": rid, "key": "model.params.max_depth", "value": str(i + 1)},
                    {"run_id": rid, "key": "seed", "value": "1"},
                ],
            )
    yield engine
    engine.dispose()


def test_optional_silver_registry_does_not_get_created(engine, tmp_path):
    from sqlalchemy import inspect

    assert not inspect(engine).has_table("silver_datasets")
    assert read_registry(engine, "silver_datasets").empty
    assert ad.load_assets(engine, tmp_path)["silver_datasets"].empty
    assert not inspect(engine).has_table("silver_datasets")


def test_launch_loader_and_layouts(engine):
    launches, runs, params = sd.load_sweep_data(engine)
    assert launches.experiment_run_id.tolist() == ["sweep1"]
    frame = sd.parameter_frame(runs, params)
    assert sd.varied_parameters(frame) == ["param::model.params.max_depth"]
    assert assets_page.layout(engine) is not None
    assert sweeps_page.layout(engine) is not None


def test_grid_keeps_dotted_parameter_names_literal():
    component = grid(pd.DataFrame({"model.params.max_depth": [3]}))
    assert component.dashGridOptions["suppressFieldDotNotation"]
    assert component.rowData[0]["model.params.max_depth"] == 3


def test_app_retains_old_and_new_callbacks(engine):
    app = create_app(engine)
    assert "run-header.children" in app.callback_map
    assert any("experiments-leaderboard" in k for k in app.callback_map)
    assert any("assets-top" in k for k in app.callback_map)
    assert any("sweeps-ranking" in k for k in app.callback_map)
    assert app.layout() is not None


def test_empty_database_layouts(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty_pages.db'}")
    ensure_tracking_tables(engine)
    try:
        assert assets_page.layout(engine) is not None
        assert sweeps_page.layout(engine) is not None
    finally:
        engine.dispose()
