import json

import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")
pytest.importorskip("sqlalchemy")
pytest.importorskip("pyarrow")

from sqlalchemy import create_engine  # noqa: E402

from ml_template.dashboard import compare_data as cd  # noqa: E402
from ml_template.dashboard import experiment_data as ed  # noqa: E402
from ml_template.dashboard.app import create_app  # noqa: E402
from ml_template.dashboard.pages import compare_page  # noqa: E402
from ml_template.db.tracking_tables import (  # noqa: E402
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_metrics,
    run_params,
    splits,
)


@pytest.fixture
def fixture(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'compare.db'}")
    ensure_tracking_tables(engine)
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"], "fold": [0, 0, 1, 1, -1]}
    ).to_parquet(split_dir / "assignments.parquet", index=False)
    silver_root = tmp_path / "silver"
    silver = silver_root / "toy" / "s"
    silver.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3, 4, 5], "target": [1, 0, 0, 1, 1]}).to_parquet(
        silver / "data.parquet", index=False
    )
    with engine.begin() as conn:
        conn.execute(
            experiment_launches.insert(),
            {
                "experiment_run_id": "launch",
                "experiment_name": "comparison",
                "kind": "sweep",
                "n_runs": 2,
                "status": "succeeded",
                "started_at": "2026-10-16T00:00:00Z",
            },
        )
        conn.execute(
            splits.insert(),
            {
                "split_id": "sp",
                "silver_dataset_version_id": "s",
                "path": str(split_dir),
                "policy_json": "{}",
                "key_col": "id",
                "target_col": "target",
            },
        )
        for i, rid in enumerate(("a", "b")):
            folder = tmp_path / rid
            folder.mkdir()
            pd.DataFrame(
                {"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "p_oof": [0.9, 0.1, 0.2, 0.8]}
            ).to_parquet(folder / "oof_predictions.parquet", index=False)
            (folder / "features.json").write_text(json.dumps({"features": ["speed", "torque"]}))
            conn.execute(
                model_runs.insert(),
                {
                    "run_id": rid,
                    "experiment_run_id": "launch",
                    "experiment_name": "comparison",
                    "config_hash": rid,
                    "status": "succeeded",
                    "seed": i,
                    "started_at": f"2026-10-16T0{i}:00:00Z",
                    "silver_dataset_version_id": "s",
                    "split_id": "sp",
                    "model_type": "xgb",
                    "primary_metric": "pr_auc",
                    "primary_value": 0.9,
                    "duration_s": 10.0,
                    "artifact_dir": str(folder),
                },
            )
            conn.execute(
                run_metrics.insert(),
                [
                    {"run_id": rid, "split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.9},
                    {
                        "run_id": rid,
                        "split": "holdout",
                        "fold": -1,
                        "metric": "pr_auc",
                        "value": 0.99,
                    },
                ],
            )
            conn.execute(run_params.insert(), {"run_id": rid, "key": "seed", "value": str(i)})
    selected = cd.selected_runs(ed.load_experiment_runs(engine), ["a", "b"])
    yield engine, selected, silver_root, tmp_path
    engine.dispose()


def test_full_bundle_and_view(fixture):
    engine, selected, silver_root, _ = fixture
    bundle = cd.load_comparison(engine, selected, silver_root=silver_root)
    assert set(bundle["predictions"]) == {"a", "b"}
    assert len(bundle["predictions"]["a"]) == 4
    assert bundle["features"]["a"] == ["speed", "torque"]
    assert cd.metric_table(bundle["metrics"], ["a", "b"]).iloc[0]["a"] == 0.9
    view = compare_page.build_view(
        bundle, "a", "pr_auc", True, None, None, None, True, "budget", 0.5, 0.05
    )
    assert view is not None


def test_missing_predictions_disable_overlay_not_metrics(fixture):
    engine, selected, silver_root, tmp_path = fixture
    (tmp_path / "b" / "oof_predictions.parquet").unlink()
    bundle = cd.load_comparison(engine, selected, silver_root=silver_root)
    assert not bundle["predictions"]
    assert not bundle["metrics"].empty
    assert any("predictions" in warning for warning in bundle["warnings"])


def test_missing_assignments_fail_closed(fixture):
    engine, selected, silver_root, tmp_path = fixture
    (tmp_path / "split" / "assignments.parquet").unlink()
    bundle = cd.load_comparison(engine, selected, silver_root=silver_root)
    assert not bundle["predictions"]
    assert any("assignments unavailable" in warning for warning in bundle["warnings"])


def test_five_tab_app_retains_all_callbacks(fixture):
    engine, _, _, _ = fixture
    app = create_app(engine)
    assert "run-header.children" in app.callback_map
    for prefix in ("experiments-leaderboard", "assets-top", "sweeps-ranking", "compare-body"):
        assert any(prefix in key for key in app.callback_map)
    assert compare_page.layout(engine) is not None
    assert app.layout() is not None


def test_empty_comparison_layout(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    ensure_tracking_tables(engine)
    try:
        assert compare_page.layout(engine) is not None
    finally:
        engine.dispose()
