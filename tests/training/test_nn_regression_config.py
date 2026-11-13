"""Task routing and configuration compatibility; no Torch import required."""

import copy

import pytest

from ml_template.models.model_factory import build_adapter_factory
from ml_template.models.nn.params import NNParams, NNRegressionParams
from ml_template.training.run_config import RunConfig, config_hash


def request():
    return {
        "task": "regression",
        "experiment_name": "nn_regression_test",
        "data": {
            "dataset_name": "test",
            "dataset_config": "unused.yaml",
            "silver_dataset_version_id": "silver_test",
        },
        "model": {"type": "nn", "params": {"hidden_dims": [8], "monitor": "mae"}},
    }


def test_nn_regression_defaults_and_factory_are_torch_free():
    cfg = RunConfig.model_validate(request())
    assert isinstance(cfg.model.params, NNRegressionParams)
    assert cfg.model.params.task == "regression"
    assert cfg.split.method == "random"
    assert cfg.analysis.learning_curve_metric == "mae"
    assert not cfg.analysis.permutation_importance
    assert not cfg.analysis.error_sample_size
    assert build_adapter_factory(cfg)().max_rounds == cfg.model.params.max_epochs
    restored = RunConfig.model_validate_json(cfg.model_dump_json())
    assert config_hash(restored) == config_hash(cfg)


def test_omitted_nn_params_get_regression_defaults():
    raw = request()
    del raw["model"]["params"]
    assert isinstance(RunConfig.model_validate(raw).model.params, NNRegressionParams)


@pytest.mark.parametrize(
    "top,nested", [("regression", "binary_classification"), ("binary_classification", "regression")]
)
def test_task_mismatch_rejected(top, nested):
    raw = request()
    raw["task"] = top
    raw["model"]["params"] = {"task": nested}
    with pytest.raises(ValueError):
        RunConfig.model_validate(raw)


def test_double_preprocessing_rejected():
    raw = request()
    raw["fitted_features"] = [
        {
            "step": "numeric_preprocessing",
            "columns": ["x"],
            "numeric_strategy": "median",
            "scaling": "standard",
        }
    ]
    with pytest.raises(ValueError, match="owns preprocessing"):
        RunConfig.model_validate(raw)


def test_binary_nn_default_params_stay_binary():
    raw = request()
    raw["task"] = "binary_classification"
    raw["model"]["params"] = {}
    cfg = RunConfig.model_validate(raw)
    assert type(cfg.model.params) is NNParams
    assert cfg.split.method == "stratified"


def test_xgb_regression_defaults_remain_supported():
    raw = request()
    raw["model"] = {"type": "xgb", "params": {"objective": "reg:squarederror"}}
    cfg = RunConfig.model_validate(raw)
    assert cfg.model.params.eval_metric == "rmse"


def test_validation_does_not_mutate_request():
    raw = request()
    before = copy.deepcopy(raw)
    RunConfig.model_validate(raw)
    assert raw == before
