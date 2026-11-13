"""Sweep keys define table rows; recorded launch runs define observations."""

import json

import pandas as pd
import pytest

from ml_template.dashboard.sweep_declared_parameters import (
    declared_parameter_frame,
    read_sweep_document,
    sweep_grid,
)


def document():
    return {
        "base": "base.yaml",
        "grid": {
            "model.params.colsample_bytree": [0.6, 0.75, 0.9],
            "model.params.learning_rate": [0.03, 0.1],
        },
    }


def inventories():
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "other"],
            "experiment_run_id": ["launch", "launch", "launch", "other-launch"],
            "primary_value": [0.9, 0.8, 0.7, 0.1],
            "status": ["succeeded"] * 4,
            "silver_dataset_version_id": ["silver"] * 4,
            "split_id": ["split"] * 4,
            "primary_metric": ["auc"] * 4,
        }
    )
    values = {
        "a": {
            "model.params.colsample_bytree": "0.6",
            "model.params.learning_rate": "0.03",
            "data.dataset_config": "wrong.yaml",
            "model.params.device": "cuda",
        },
        "b": {
            "model.params.colsample_bytree": "0.6",
            "model.params.learning_rate": "0.1",
            "data.dataset_config": "wrong.yaml",
            "model.params.device": "cuda",
        },
        "c": {
            "model.params.colsample_bytree": "0.9",
            "model.params.learning_rate": "0.03",
            "data.dataset_config": "wrong.yaml",
            "model.params.device": "cuda",
        },
        "other": {"model.params.colsample_bytree": "0.6", "model.params.learning_rate": "0.03"},
    }
    params = pd.DataFrame(
        [
            {"run_id": run, "key": key, "value": value}
            for run, snapshot in values.items()
            for key, value in snapshot.items()
        ]
    )
    return runs, params


def test_only_declared_keys_not_all_model_or_dataset_config():
    runs, params = inventories()
    frame = declared_parameter_frame(runs, params, document(), launch_id="launch")
    assert {c for c in frame if c.startswith("param::")} == {
        "param::model.params.colsample_bytree",
        "param::model.params.learning_rate",
    }
    assert set(frame["run_id"]) == {"a", "b", "c"}


def test_other_swept_settings_remain_unrestricted():
    runs, params = inventories()
    frame = declared_parameter_frame(runs, params, document(), launch_id="launch")
    part = frame[frame["param::model.params.colsample_bytree"].eq("0.6")]
    assert part["primary_value"].mean() == pytest.approx(0.85)
    assert part["primary_value"].std() == pytest.approx(0.1 / 2**0.5)
    assert set(part["param::model.params.learning_rate"]) == {"0.03", "0.1"}


def test_yaml_candidate_values_do_not_replace_recorded_values():
    runs, params = inventories()
    spec = document()
    spec["grid"]["model.params.learning_rate"] = [999]
    frame = declared_parameter_frame(runs, params, spec, launch_id="launch")
    assert "999" not in set(frame["param::model.params.learning_rate"])


def test_declared_singleton_included():
    runs, params = inventories()
    spec = document()
    spec["grid"]["model.params.device"] = ["cuda"]
    result = declared_parameter_frame(runs, params, spec, launch_id="launch")
    assert result["param::model.params.device"].tolist() == ["cuda"] * 3


def test_seed_grid_uses_raw_run_params_despite_general_frame_exclusions():
    runs, params = inventories()
    extra = pd.DataFrame({"run_id": ["a", "b", "c"], "key": ["seed"] * 3, "value": ["1", "2", "3"]})
    spec = {"base": "base.yaml", "grid": {"seed": [1, 2, 3]}}
    result = declared_parameter_frame(runs, pd.concat([params, extra]), spec, launch_id="launch")
    assert result["param::seed"].tolist() == ["1", "2", "3"]


def test_missing_declared_parameter_is_not_fabricated_from_yaml():
    runs, params = inventories()
    params = params[~((params["run_id"] == "b") & (params["key"] == "model.params.learning_rate"))]
    result = declared_parameter_frame(runs, params, document(), launch_id="launch")
    assert result.loc[result["run_id"].eq("b"), "param::model.params.learning_rate"].isna().all()


def test_duplicate_param_key_rejected():
    runs, params = inventories()
    with pytest.raises(ValueError, match="Duplicate"):
        declared_parameter_frame(
            runs, pd.concat([params, params.iloc[:1]]), document(), launch_id="launch"
        )


def test_no_sweep_path_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="sweep_path"):
        read_sweep_document(None, repo_root=tmp_path)


def test_missing_yaml_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_sweep_document("missing.yaml", repo_root=tmp_path)


def test_reads_sweep_not_base_or_dataset_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "configs/sweep.yaml"
    path.parent.mkdir()
    path.write_text(json.dumps(document()))
    # base.yaml does not exist: loading the sweep must still work.
    spec, source = read_sweep_document("configs/sweep.yaml", repo_root=tmp_path)
    assert set(spec["grid"]) == set(document()["grid"])
    assert source == path.resolve()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"base": "x", "grid": {}},
        {"base": "x", "grid": {"x": []}},
        {"base": "x", "grid": {"x": 1}},
        {"base": "x", "grid": {"param::x": [1]}},
        {"base": "x", "grid": {"x": [1]}, "other": 1},
    ],
)
def test_bad_sweep_yaml_rejected(bad):
    with pytest.raises(ValueError):
        sweep_grid(bad)


def test_entire_launch_keeps_other_cohorts_for_explicit_validation():
    runs, params = inventories()
    runs.loc[1, "split_id"] = "other-split"
    result = declared_parameter_frame(runs, params, document(), launch_id="launch")
    assert len(result) == 3
    assert result["split_id"].nunique() == 2


def test_empty_parameter_table_keeps_runs_and_missing_columns():
    runs, _ = inventories()
    result = declared_parameter_frame(
        runs, pd.DataFrame(columns=["run_id", "key", "value"]), document(), launch_id="launch"
    )
    assert len(result) == 3
    assert result["param::model.params.learning_rate"].isna().all()


def test_invalid_yaml_syntax_is_actionable(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("grid: [unclosed")
    with pytest.raises(ValueError, match="Invalid sweep YAML"):
        read_sweep_document(path, repo_root=tmp_path)


def test_full_loader_uses_recorded_sweep_path_and_whole_launch(tmp_path, monkeypatch):
    from ml_template.dashboard.sweep_declared_parameters import load_launch_summary_inputs

    runs, params = inventories()
    spec = document()
    spec["experiment_name"] = "my-sweep"
    source = tmp_path / "sweep.yaml"
    source.write_text(json.dumps(spec))
    launches = pd.DataFrame(
        {
            "experiment_run_id": ["launch"],
            "experiment_name": ["my-sweep"],
            "sweep_path": [str(source)],
        }
    )
    monkeypatch.setattr(
        "ml_template.dashboard.sweep_data.load_sweep_data",
        lambda engine: (launches, runs, params),
    )
    monkeypatch.setattr(
        "ml_template.paths.ROOT",
        tmp_path,
    )
    result, actual_spec, actual_source = load_launch_summary_inputs(None, "launch")
    assert len(result) == 3
    assert set(actual_spec["grid"]) == set(document()["grid"])
    assert actual_source == str(source.resolve())
    assert "param::data.dataset_config" not in result


def test_current_yaml_name_mismatch_is_not_silently_used(tmp_path, monkeypatch):
    from ml_template.dashboard.sweep_declared_parameters import load_launch_summary_inputs

    runs, params = inventories()
    spec = document()
    spec["experiment_name"] = "changed-name"
    source = tmp_path / "sweep.yaml"
    source.write_text(json.dumps(spec))
    launches = pd.DataFrame(
        {
            "experiment_run_id": ["launch"],
            "experiment_name": ["original-name"],
            "sweep_path": [str(source)],
        }
    )
    monkeypatch.setattr(
        "ml_template.dashboard.sweep_data.load_sweep_data",
        lambda engine: (launches, runs, params),
    )
    monkeypatch.setattr(
        "ml_template.paths.ROOT",
        tmp_path,
    )
    with pytest.raises(ValueError, match="differs"):
        load_launch_summary_inputs(None, "launch")
