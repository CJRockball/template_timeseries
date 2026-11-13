import numpy as np
import pytest
from sqlalchemy import create_engine

pytest.importorskip("pyarrow")
pytest.importorskip("dash_ag_grid")

from ml_template.dashboard import run_figs as rf  # noqa: E402
from ml_template.dashboard.app import create_app  # noqa: E402
from ml_template.dashboard.pages import run_page as page  # noqa: E402
from ml_template.db.tracking_tables import (  # noqa: E402
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_feature_importance,
    run_metrics,
)


def _run(run_id, started, **extra):
    row = {
        "run_id": run_id,
        "experiment_run_id": "launch1",
        "experiment_name": "exp",
        "config_hash": "h",
        "status": "succeeded",
        "started_at": started,
        "silver_dataset_version_id": "s",
        "split_id": "sp",
        "model_type": "xgb",
        "seed": 1,
        "primary_metric": "pr_auc",
        "primary_value": 0.8,
        "n_rounds_final": 50,
        "duration_s": 25.0,
    }
    return {**row, **extra}


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'page.db'}")
    ensure_tracking_tables(eng)
    with eng.begin() as conn:
        conn.execute(
            experiment_launches.insert(),
            [
                {
                    "experiment_run_id": "launch1",
                    "experiment_name": "exp",
                    "kind": "single",
                    "n_runs": 2,
                    "status": "succeeded",
                    "started_at": "2026-10-15T00:00:00Z",
                }
            ],
        )
        conn.execute(
            model_runs.insert(),
            [
                _run("r1", "2026-10-15T01:00:00Z"),
                _run("r2", "2026-10-15T03:00:00Z", holdout_evaluated_at="2026-10-15T04:00:00Z"),
            ],
        )
        conn.execute(
            run_metrics.insert(),
            [
                {"run_id": "r2", "split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.8},
                {"run_id": "r2", "split": "cv", "fold": 0, "metric": "pr_auc", "value": 0.7},
                {"run_id": "r2", "split": "cv", "fold": 1, "metric": "pr_auc", "value": 0.9},
            ],
        )
        conn.execute(
            run_feature_importance.insert(),
            [
                {
                    "run_id": "r2",
                    "split": "cv",
                    "fold": -1,
                    "method": "permutation",
                    "feature": "a",
                    "value": 0.2,
                    "std": 0.01,
                },
            ],
        )
    return eng


def test_default_run_is_latest(engine):
    options, default = page.run_choices(engine, page.ALL)
    assert default == "r2" and len(options) == 2


def test_no_runs_is_handled(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    ensure_tracking_tables(eng)
    assert page.run_choices(eng, page.ALL) == ([], None)
    assert page.layout(eng) is not None
    assert page.metrics_view(eng, None)[1] is None


def test_metrics_view_defaults_to_primary_metric(engine):
    metrics, default, _ = page.metrics_view(engine, "r2")
    assert metrics == ["pr_auc"] and default == "pr_auc"


def test_missing_artifacts_give_placeholders(engine):
    lc, gap = page.curves_view(engine, "r2", overlay=False)
    assert lc.layout.annotations and gap.layout.annotations
    pr, rel = page.scoring_figs(engine, "r2")
    assert pr.layout.annotations and rel.layout.annotations


def test_importance_methods_and_cost(engine):
    assert page.importance_methods(engine, "r2") == (["permutation"], "permutation")
    fig, text = page.cost_view(engine, "r2")
    assert "0.500 s per round" in text and len(fig.data) == 2


def test_holdout_label():
    assert page.holdout_label("2026-10-15T04:00:00Z") == "holdout looked at 2026-10-15T04:00:00Z"
    assert page.holdout_label(None) == "holdout untouched"
    assert page.holdout_label(float("nan")) == "holdout untouched"


def test_header_view_builds(engine):
    assert page.header_view(engine, "r2") is not None
    assert page.header_view(engine, None) is not None


def test_app_builds_with_callbacks(engine):
    app = create_app(engine)
    assert len(app.callback_map) >= 10
    assert app.layout() is not None


def test_threshold_summary_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    s = rf.threshold_summary(y, p, 0.5)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 2)
    assert s["precision"] == pytest.approx(0.5) and s["alert_rate"] == pytest.approx(0.4)
