import pandas as pd
import pytest

from ml_template.dashboard import sweep_data as sd
from ml_template.dashboard.sweep_figs import parameter_plot

X = "param::model.params.max_depth"
Y = "param::model.params.min_child_weight"
Z = "param::model.params.learning_rate"


def _data():
    rows = []
    for i, (depth, weight, rate, seed, score) in enumerate(
        [
            (2, 1, 0.05, 1, 0.70),
            (2, 1, 0.05, 2, 0.80),
            (2, 1, 0.05, 1, 0.72),
            (3, 1, 0.05, 1, 0.85),
            (2, 5, 0.05, 1, 0.75),
            (2, 1, 0.10, 1, 0.99),
        ]
    ):
        rows.append(
            {
                "run_id": f"r{i}",
                "experiment_run_id": "launch",
                "experiment_name": "e",
                "silver_dataset_version_id": "s",
                "split_id": "sp",
                "primary_metric": "pr_auc",
                "primary_value": score,
                "seed": seed,
                "duration_s": 10.0,
                "status": "succeeded",
                "started_at": f"2026-10-16T0{i}:00:00Z",
                "params_recorded": True,
                "git_sha": "abc",
                "git_dirty": 0,
                X: str(depth),
                Y: str(weight),
                Z: str(rate),
            }
        )
    return pd.DataFrame(rows)


def _fixed():
    frame, _ = sd.context_groups(_data(), [X, Y])
    context = frame.loc[frame[Z] == "0.05", "context_id"].iloc[0]
    return frame[frame.context_id == context]


def test_varied_axes():
    assert sd.varied_parameters(_data()) == sorted([X, Y, Z])


def test_context_does_not_pool_other_learning_rates():
    frame, options = sd.context_groups(_data(), [X, Y])
    assert len(options) == 2
    with pytest.raises(ValueError, match="fixed-parameter"):
        sd.cell_summary(frame, [X, Y])


def test_seed_cell_summary_deduplicates_latest():
    summary = sd.cell_summary(_fixed(), [X, Y])
    cell = summary[(summary[X] == "2") & (summary[Y] == "1")].iloc[0]
    assert cell["mean"] == pytest.approx(0.76)
    assert cell["n_seeds"] == 2
    assert cell["std"] == pytest.approx(pd.Series([0.72, 0.80]).std())


def test_failed_run_and_null_seed_excluded():
    frame = _fixed()
    frame.loc[frame.run_id == "r1", "status"] = "failed"
    frame.loc[frame.run_id == "r2", "seed"] = None
    summary = sd.cell_summary(frame, [X, Y])
    cell = summary[(summary[X] == "2") & (summary[Y] == "1")].iloc[0]
    assert cell["mean"] == pytest.approx(0.70)
    assert cell["n_seeds"] == 1 and pd.isna(cell["std"])


def test_duplicate_axes_and_mixed_cohorts_rejected():
    with pytest.raises(ValueError, match="different"):
        sd.cell_summary(_fixed(), [X, X])
    frame = _fixed()
    frame.loc[frame.run_id == "r1", "split_id"] = "other"
    with pytest.raises(ValueError, match="dataset"):
        sd.cell_summary(frame, [X, Y])


def test_missing_parameter_records_isolate_contexts():
    frame = _data()
    frame["params_recorded"] = False
    _, options = sd.context_groups(frame, [X, Y])
    assert len(options) == len(frame)


def test_parameter_pivot_excludes_model_seed_not_split_seed():
    frame = _data().drop(columns=[X, Y, Z])
    params = pd.DataFrame(
        [
            {"run_id": "r0", "key": "seed", "value": "1"},
            {"run_id": "r0", "key": "split.seed", "value": "42"},
            {"run_id": "r0", "key": "analysis.learning_curves", "value": "True"},
            {"run_id": "r0", "key": "model.params.max_depth", "value": "2"},
        ]
    )
    wide = sd.parameter_frame(frame, params)
    assert X in wide and "param::split.seed" in wide
    assert "param::seed" not in wide and "param::analysis.learning_curves" not in wide


def test_launch_progress_not_registered_and_duration():
    launches = pd.DataFrame(
        [{"experiment_run_id": "launch", "n_runs": 8, "started_at": "2026-10-16T00:00:00Z"}]
    )
    summary = sd.launch_summary(launches, _data()).iloc[0]
    assert summary["not_registered"] == 2
    assert summary["registered_runs"] == 6
    assert summary["sum_recorded_duration_s"] == 60


def test_empty_launch_progress():
    launches = pd.DataFrame(columns=["experiment_run_id", "n_runs", "started_at"])
    assert sd.launch_summary(launches, _data()).empty


def test_one_parameter_plot():
    frame, _ = sd.context_groups(_data(), [X])
    context = frame.loc[(frame[Y] == "1") & (frame[Z] == "0.05"), "context_id"].iloc[0]
    summary = sd.cell_summary(frame[frame.context_id == context], [X])
    fig = parameter_plot(summary, [X])
    assert len(fig.data) == 1 and len(fig.data[0].x) == 2


def test_heatmap_leaves_unobserved_cell_blank():
    summary = sd.cell_summary(_fixed(), [X, Y])
    fig = parameter_plot(summary, [X, Y])
    assert fig.data[0].type == "heatmap"
    assert pd.isna(fig.data[0].z).sum() == 1


def test_no_axes_placeholder():
    assert parameter_plot(pd.DataFrame(), []).layout.annotations


def test_code_revision_changes_context():
    frame = _data()
    frame.loc[0, "git_sha"] = "changed"
    _, options = sd.context_groups(frame, [X, Y])
    assert len(options) == 3
