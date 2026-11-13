import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")
pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine  # noqa: E402

from ml_template.dashboard import experiment_data as ed  # noqa: E402
from ml_template.dashboard.app import create_app  # noqa: E402
from ml_template.dashboard.pages import experiment_page as page  # noqa: E402
from ml_template.db.tracking_tables import (  # noqa: E402
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_params,
)


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'experiment.db'}")
    ensure_tracking_tables(engine)
    with engine.begin() as conn:
        conn.execute(
            experiment_launches.insert(),
            {
                "experiment_run_id": "launch",
                "experiment_name": "baseline",
                "kind": "sweep",
                "n_runs": 2,
                "status": "succeeded",
                "started_at": "2026-10-16T00:00:00Z",
            },
        )
        for seed in (1, 2):
            rid = f"run{seed}"
            conn.execute(
                model_runs.insert(),
                {
                    "run_id": rid,
                    "experiment_run_id": "launch",
                    "experiment_name": "baseline",
                    "config_hash": str(seed),
                    "status": "succeeded",
                    "started_at": f"2026-10-16T0{seed}:00:00Z",
                    "silver_dataset_version_id": "silver",
                    "split_id": "split",
                    "model_type": "xgb",
                    "seed": seed,
                    "primary_metric": "pr_auc",
                    "primary_value": 0.7 + seed / 100,
                    "duration_s": 10.0,
                },
            )
            conn.execute(
                run_params.insert(),
                [
                    {"run_id": rid, "key": "seed", "value": str(seed)},
                    {"run_id": rid, "key": "model.params.max_depth", "value": "3"},
                ],
            )
    yield engine
    engine.dispose()


def test_database_loader_and_page(engine):
    runs = ed.load_experiment_runs(engine)
    assert len(runs) == 2
    cohort = ed.cohort_options(runs)[0]["value"]
    table, fig, detail, note = page.build_view(engine, cohort, [], "higher")
    assert table.rowData[0]["n_seeds"] == 2
    assert len(fig.data) == 1
    assert len(detail.rowData) == 2
    assert "1 experiment/recipe" in note
    assert page.layout(engine) is not None


def test_seed_plot_has_run_ids(engine):
    runs = ed.load_experiment_runs(engine)
    fig = page.seed_figure(ed.seed_rows(runs), ed.seed_leaderboard(runs))
    assert set(fig.data[0].customdata[:, 1]) == {"run1", "run2"}


def test_empty_selection_is_safe(engine):
    _, fig, detail, _ = page.build_view(engine, None, [], "higher")
    assert not fig.data
    assert not detail.rowData


def test_new_app_keeps_single_run_callbacks(engine):
    app = create_app(engine)
    assert "run-header.children" in app.callback_map
    assert any("experiments-leaderboard" in key for key in app.callback_map)
    assert app.layout() is not None
