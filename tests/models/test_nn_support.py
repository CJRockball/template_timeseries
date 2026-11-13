import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_template.models.nn.data import batch_indices, binary_labels, resolve_pos_weight
from ml_template.models.nn.metrics import EarlyStop, metric_value, probabilities
from ml_template.models.nn.params import NNModelConfig, NNParams
from ml_template.models.nn_classifier import NNAdapter
from ml_template.training.nn_smoke import SmokeConfig


def test_cpu_defaults_and_lazy_torch_import():
    p = NNParams()
    assert p.device == p.inference_device == "cpu"
    assert NNModelConfig().type == "nn"
    assert NNAdapter(p, 42).max_rounds == p.max_epochs
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ml_template.models.nn_classifier import NNAdapter; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        {"device": "auto"},
        {"device": "gpu"},
        {"architecture": "ft_transformer"},
        {"task": "regression"},
        {"task": "multiclass_classification"},
        {"max_epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"hidden_dims": []},
        {"hidden_dims": [4, 0]},
        {"normalization": "batchnorm"},
        {"dropout": 1},
        {"embedding_dropout": -1},
        {"pos_weight": 0},
        {"pos_weight": "bad"},
        {"learning_rate": float("nan")},
        {"unknown": 1},
        {"preprocessing": {"numeric": [], "categorical": []}},
        {"preprocessing": {"numeric": ["x"], "categorical": ["x"]}},
    ],
)
def test_invalid_params(raw):
    with pytest.raises(ValidationError):
        NNParams.model_validate(raw)


@pytest.mark.parametrize("n,batch", [(0, 4), (1, 1), (5, 2), (4, 10), (101, 8)])
def test_batch_coverage(n, batch):
    parts = list(batch_indices(n, batch, np.random.default_rng(42)))
    flat = np.concatenate(parts) if parts else np.array([], dtype=int)
    assert sorted(flat) == list(range(n))
    assert all(0 < len(p) <= batch for p in parts)


def test_batch_reproducibility_and_remainder():
    a = list(batch_indices(9, 4, np.random.default_rng(42)))
    b = list(batch_indices(9, 4, np.random.default_rng(42)))
    assert [len(p) for p in a] == [4, 4, 1]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kind", ["nan", "multiclass", "unaligned", "one_class"])
def test_label_errors(kind):
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    if kind == "nan":
        y.iloc[0] = np.nan
    elif kind == "multiclass":
        y.iloc[0] = 2
    elif kind == "unaligned":
        y.index = [4, 5, 6, 7]
    else:
        y[:] = 1
    with pytest.raises(ValueError):
        binary_labels(X, y)


def test_binary_labels_and_weight():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    values = binary_labels(X, pd.Series([0, 1, 0, 0]))
    assert values.dtype == np.float32
    assert resolve_pos_weight("auto", values) == 3
    assert resolve_pos_weight(1, values) == 1


@pytest.mark.parametrize(
    "monitor,values",
    [
        ("logloss", [0.8, 0.5, 0.6, 0.7]),
        ("auc", [0.5, 0.9, 0.8, 0.7]),
        ("aucpr", [0.2, 0.6, 0.6, 0.6]),
    ],
)
def test_early_stop_direction(monitor, values):
    stop = EarlyStop(2, monitor)
    assert [stop.update(v, i) for i, v in enumerate(values, 1)] == [True, True, False, False]
    assert stop.best_epoch == 2 and stop.stopped


def test_min_delta_and_nonfinite_metric():
    stop = EarlyStop(2, "auc", 0.05)
    assert stop.update(0.5, 1)
    assert not stop.update(0.52, 2)
    with pytest.raises(ValueError):
        stop.update(float("nan"), 3)


@pytest.mark.parametrize("name", ["logloss", "auc", "aucpr", "error"])
def test_metrics(name):
    assert np.isfinite(metric_value(name, np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])))


def test_sigmoid_stability_and_shape():
    np.testing.assert_allclose(probabilities([-10000, 0, 10000]), [0, 0.5, 1])
    with pytest.raises(ValueError):
        probabilities([np.nan])
    with pytest.raises(ValueError):
        probabilities([[0]])


def test_unfitted_and_bad_seed():
    with pytest.raises(ValueError):
        NNAdapter(NNParams(), -1)
    with pytest.raises(RuntimeError):
        NNAdapter(NNParams(), 42).predict_proba(pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"source": "synthetic", "model_file": "m.yaml", "experiment": "e.yaml"},
        {"source": "experiment", "model_file": "m.yaml"},
        {"source": "synthetic", "model_file": "m.yaml", "sample_rows": 1},
        {"source": "synthetic", "model_file": "m.yaml", "timeout_seconds": 0},
    ],
)
def test_smoke_validation(raw):
    with pytest.raises(ValidationError):
        SmokeConfig.model_validate(raw)


def test_adapter_ownership_with_explicit_fake_backend(monkeypatch):
    class FakePreprocessor:
        def __init__(self, config):
            self.config = config

        def fit(self, X):
            self.rows = X.index.tolist()
            return self

        def transform_arrays(self, X):
            return SimpleNamespace(
                numeric=np.zeros((len(X), 2), dtype=np.float32),
                categorical=np.empty((len(X), 0), dtype=np.int64),
                cardinalities=(),
            )

    parent = ModuleType("ml_template.preprocessing")
    parent.__path__ = []
    module = ModuleType("ml_template.preprocessing.nn_preprocessor")
    module.NNPreprocessor = FakePreprocessor
    monkeypatch.setitem(sys.modules, "ml_template.preprocessing", parent)
    monkeypatch.setitem(sys.modules, "ml_template.preprocessing.nn_preprocessor", module)
    monkeypatch.setattr(
        "ml_template.models.nn_classifier.resolve_preprocessing", lambda p, X: object()
    )
    calls = []

    def train(inputs, labels, params, seed, epochs, **kwargs):
        calls.append((len(labels), epochs, kwargs["early_stopping"]))
        return SimpleNamespace(model=object(), best_epoch=2, history=[], execution={})

    backend = SimpleNamespace(
        train=train, predict_logits=lambda m, inp, p, s: np.zeros(len(inp.numeric))
    )
    monkeypatch.setattr("ml_template.models.nn_classifier._engine", lambda: backend)
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    a = NNAdapter(NNParams(max_epochs=3), 42)
    assert a.fit_with_validation(X.iloc[:4], y.iloc[:4], X.iloc[4:], y.iloc[4:]) == 2
    assert a.preprocessor.rows == [0, 1, 2, 3]
    a.predict_proba(X.iloc[4:])
    b = NNAdapter(NNParams(max_epochs=3), 42)
    b.refit(X, y, 2)
    assert a.preprocessor is not b.preprocessor and b.preprocessor.rows == list(range(6))
    assert calls == [(4, 3, True), (6, 2, False)]
