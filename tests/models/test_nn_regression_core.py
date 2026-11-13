"""Torch-free continuous-target and schema acceptance."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_template.models.nn.params import NNModelConfig, NNParams, NNRegressionParams
from ml_template.models.nn.regression import RegressionStop, metric_value, targets


def test_default_binary_nn_schema_is_unchanged():
    config = NNModelConfig()
    assert type(config.params) is NNParams
    assert config.params.task == "binary_classification"
    assert "loss" not in config.params.model_dump()
    assert NNModelConfig.model_validate_json(config.model_dump_json()) == config


def test_regression_schema_roundtrip():
    config = NNModelConfig(params={"task": "regression", "monitor": "mae"})
    assert type(config.params) is NNRegressionParams
    assert config.params.loss == "mse"
    assert NNModelConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "override",
    [
        {"monitor": "logloss"},
        {"loss": "bce"},
        {"pos_weight": "auto"},
        {"pos_weight": 2.0},
        {"task": "binary_classification"},
    ],
)
def test_regression_rejects_binary_options(override):
    with pytest.raises(ValidationError):
        NNRegressionParams(**override)


@pytest.mark.parametrize("values", [[-2.5, 0.25, 10.75], [2.0, 2.0, 2.0]])
def test_continuous_or_constant_targets_are_valid(values):
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = targets(X, pd.Series(values))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, values)


@pytest.mark.parametrize(
    "values",
    [
        [True, False, True],
        ["1", "2", "3"],
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
        [1.0, -np.inf, 3.0],
        [1.0, 1e100, 3.0],
        [1j, 2j, 3j],
    ],
)
def test_invalid_targets_rejected(values):
    with pytest.raises(ValueError):
        targets(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), pd.Series(values))


def test_nullable_numeric_target_dtype_with_no_nulls():
    result = targets(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([1.25, -2.5], dtype="Float64"))
    np.testing.assert_allclose(result, [1.25, -2.5])


def test_target_alignment_required():
    with pytest.raises(ValueError):
        targets(pd.DataFrame({"x": [1.0, 2.0]}, index=[1, 2]), pd.Series([1.0, 2.0], index=[2, 1]))


@pytest.mark.parametrize("name, expected", [("mse", 2.5), ("rmse", np.sqrt(2.5)), ("mae", 1.5)])
def test_regression_metrics(name, expected):
    assert metric_value(name, [-1.0, 2.0], [0.0, 4.0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([True], [1.0]),
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
        ([[1.0]], [[1.0]]),
    ],
)
def test_metric_rejects_bad_vectors(a, b):
    with pytest.raises(ValueError):
        metric_value("rmse", a, b)


def test_stopping_uses_minimum_and_delta():
    stopper = RegressionStop(2, 0.1)
    assert stopper.update(5.0, 1)
    assert not stopper.update(4.95, 2)
    assert stopper.update(4.0, 3)
    assert not stopper.update(4.1, 4)
    assert not stopper.stopped
    assert not stopper.update(4.2, 5)
    assert stopper.stopped and stopper.best_epoch == 3


@pytest.mark.parametrize("patience,delta", [(0, 0.0), (True, 0.0), (1, -1.0), (1, np.inf)])
def test_invalid_stopping_settings(patience, delta):
    with pytest.raises(ValueError):
        RegressionStop(patience, delta)
