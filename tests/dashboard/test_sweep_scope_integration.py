"""End-to-end pure summary checks for YAML scope and whole-launch metrics."""

import pandas as pd
import pytest

from ml_template.dashboard.sweep_declared_parameters import declared_parameter_frame
from ml_template.dashboard.sweep_parameter_summary import parameter_summaries


def sample():
    rows, parameters = [], []
    for index in range(54):
        run = f"r{index:02d}"
        rows.append(
            {
                "run_id": run,
                "experiment_run_id": "sweep",
                "status": "succeeded",
                "primary_value": 0.9 + index / 10000,
                "silver_dataset_version_id": "silver",
                "split_id": "split",
                "primary_metric": "auc",
                "params_recorded": True,
            }
        )
        for key, value in {
            "model.params.colsample_bytree": ["0.6", "0.75", "0.9"][index % 3],
            "model.params.learning_rate": ["0.03", "0.1"][index % 2],
            "data.dataset_config": "dataset.yaml",
            "model.params.device": "cuda",
        }.items():
            parameters.append({"run_id": run, "key": key, "value": value})
    spec = {
        "base": "base.yaml",
        "grid": {
            "model.params.colsample_bytree": [0.6, 0.75, 0.9],
            "model.params.learning_rate": [0.03, 0.1],
        },
    }
    return pd.DataFrame(rows), pd.DataFrame(parameters), spec


def test_all_54_runs_grouped_by_only_yaml_axes():
    runs, params, spec = sample()
    frame = declared_parameter_frame(runs, params, spec, launch_id="sweep")
    overall, top, meta = parameter_summaries(frame)
    assert meta["eligible_runs"] == 54 and meta["selected_runs"] == 10
    assert len(top) == 2
    assert len(overall) == 5
    cs = overall[overall["parameter"].eq("model.params.colsample_bytree")]
    assert cs["n_runs"].tolist() == [18, 18, 18]
    lr = overall[overall["parameter"].eq("model.params.learning_rate")]
    assert lr["n_runs"].tolist() == [27, 27]
    observed = runs.loc[[i for i in range(54) if i % 3 == 0], "primary_value"]
    actual = cs[cs["value"].eq("0.6")].iloc[0]
    assert actual["mean"] == pytest.approx(observed.mean())
    assert actual["std"] == pytest.approx(observed.std(ddof=1))


def test_failed_missing_primary_metric_does_not_block_successful_averages():
    runs, params, spec = sample()
    runs.loc[0, ["status", "primary_value", "primary_metric"]] = ["failed", None, None]
    frame = declared_parameter_frame(runs, params, spec, launch_id="sweep")
    _, _, meta = parameter_summaries(frame)
    assert meta["registered_runs"] == 54
    assert meta["eligible_runs"] == 53 and meta["excluded_runs"] == 1


def test_whole_launch_rejects_mixed_successful_cohorts_not_silent_subset():
    runs, params, spec = sample()
    runs.loc[0, "split_id"] = "other"
    frame = declared_parameter_frame(runs, params, spec, launch_id="sweep")
    with pytest.raises(ValueError, match="whole launch"):
        parameter_summaries(frame)


def test_top_k_uses_entire_launch_and_respects_direction():
    runs, params, spec = sample()
    frame = declared_parameter_frame(runs, params, spec, launch_id="sweep")
    _, high, _ = parameter_summaries(frame, k=1)
    _, low, _ = parameter_summaries(frame, k=1, higher_is_better=False)
    assert high[high["parameter"].eq("model.params.learning_rate")].iloc[0]["mode_values"] == [
        "0.1"
    ]
    assert low[low["parameter"].eq("model.params.learning_rate")].iloc[0]["mode_values"] == ["0.03"]
