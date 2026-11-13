# ruff: noqa: E402
import importlib.util

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="install the nn extra for real NN tests")
try:
    pre_spec = importlib.util.find_spec("ml_template.preprocessing.nn_preprocessor")
except ModuleNotFoundError:
    pre_spec = None
if pre_spec is None:
    pytest.skip("install the previous preprocessing bundle first", allow_module_level=True)

from ml_template.models.nn.backbones import build_backbone
from ml_template.models.nn.data import resolve_preprocessing
from ml_template.models.nn.params import NNParams
from ml_template.models.nn_classifier import NNAdapter


def frame():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"x": rng.normal(size=32), "c": ["A", "B", None, "C"] * 8})
    X.loc[::7, "x"] = np.nan
    return X, pd.Series([0, 1, 0, 1] * 8)


def params(**kwargs):
    return NNParams(
        max_epochs=3,
        patience=2,
        hidden_dims=(8, 4),
        batch_size=7,
        dropout=0.0,
        embedding_dropout=0.0,
        **kwargs,
    )


@pytest.mark.parametrize("nnum,cards", [(3, ()), (0, (3, 4)), (3, (3,))])
def test_forward_backward_singleton_branches(nnum, cards):
    model = build_backbone(nnum, cards, params())
    out = model(torch.zeros((1, nnum)), torch.zeros((1, len(cards)), dtype=torch.long))
    assert out.shape == (1,)
    torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones(1)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("branch", ["mixed", "numeric", "categorical"])
def test_train_refit_save_reload_parity(tmp_path, branch):
    X, y = frame()
    X = X[["x"]] if branch == "numeric" else X[["c"]] if branch == "categorical" else X
    a = NNAdapter(params(), 42)
    best = a.fit_with_validation(X.iloc[:24], y.iloc[:24], X.iloc[24:], y.iloc[24:])
    assert 1 <= best <= 3 and a.preprocessor.state_dict()["fit_rows"] == 24
    assert next(a.model.parameters()).device.type == "cpu"
    b = NNAdapter(params(), 42)
    b.refit(X, y, best)
    assert b.preprocessor.state_dict()["fit_rows"] == 32
    expected = b.predict_proba(X)
    assert expected.shape == (32,) and np.isfinite(expected).all()
    path = tmp_path / "model.ubj"
    b.save(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert "preprocessor" in raw and "state_dict" in raw
    loaded = NNAdapter.load(params(), 42, path)
    np.testing.assert_allclose(expected, loaded.predict_proba(X), rtol=0, atol=1e-6)
    chunks = np.concatenate([loaded.predict_proba(X.iloc[:1]), loaded.predict_proba(X.iloc[1:])])
    np.testing.assert_allclose(expected, chunks, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        expected, loaded.predict_proba(X.iloc[::-1])[::-1], rtol=0, atol=1e-6
    )
    before = loaded.preprocessor.state_dict()
    loaded.predict_proba(X)
    assert loaded.preprocessor.state_dict() == before
    with pytest.raises(ValueError, match="configuration/seed"):
        NNAdapter.load(params(), 9, path)


def test_cpu_stays_on_cpu_without_lazy_cuda_init(monkeypatch):
    from ml_template.models.nn import engine

    X, y = frame()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU execution must not initialize CUDA")

    # Our device selector must not query CUDA when CPU is requested.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", forbidden)
        assert engine.requested_device("cpu").type == "cpu"

    # Allow PyTorch's internal availability queries during optimisation.
    # Still forbid Python-level lazy CUDA initialization.
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)

    original_selector = engine.requested_device
    requested = []

    def cpu_only(name):
        requested.append(name)
        assert name == "cpu"
        return original_selector(name)

    monkeypatch.setattr(engine, "requested_device", cpu_only)

    adapter = NNAdapter(params(), 1)
    adapter.refit(X, y, 1)
    predictions = adapter.predict_proba(X)

    assert requested
    assert all(name == "cpu" for name in requested)
    assert all(p.device.type == "cpu" for p in adapter.model.parameters())
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_rng_and_thread_restore():
    X, y = frame()
    before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    a, b = NNAdapter(params(), 8), NNAdapter(params(), 8)
    a.refit(X, y, 2)
    b.refit(X, y, 2)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    assert torch.equal(torch.get_rng_state(), before) and torch.get_num_threads() == threads


@pytest.mark.parametrize("metric", ["logloss", "auc", "aucpr", "error"])
def test_learning_curve(metric):
    X, y = frame()
    a = NNAdapter(params(), 2)
    table = a.learning_curve(X.iloc[:24], y.iloc[:24], X.iloc[24:], y.iloc[24:], 2, metric)
    assert set(table.dataset) == {"train", "val"} and table.metric.eq(metric).all()
    assert len(table) == 4 and a.model is None


@pytest.mark.parametrize("branch", ["numeric", "categorical", "integer_cat"])
def test_role_resolution(branch):
    X = pd.DataFrame({"x": [1.0, 2.0], "r": [1, 2]})
    override = (
        {"categorical": []}
        if branch == "numeric"
        else {"numeric": []}
        if branch == "categorical"
        else {"categorical": ["r"]}
    )
    config = resolve_preprocessing(NNParams(preprocessing=override), X)
    assert set(config.input_columns) == set(X.columns)
    if branch == "integer_cat":
        assert config.categorical == ("r",)


def test_actual_factory_and_cv(tmp_path):
    from ml_template.models.model_factory import build_adapter_factory, load_adapter
    from ml_template.training.cross_validation import cross_validate
    from ml_template.training.run_config import RunConfig

    X, y = frame()
    cfg = RunConfig.model_validate(
        {
            "experiment_name": "nn_fixture",
            "seed": 2,
            "data": {
                "dataset_name": "fixture",
                "dataset_config": "fixture.yaml",
                "silver_dataset_version_id": "fixture",
            },
            "split": {"n_folds": 2},
            "model": {"type": "nn", "params": params().model_dump()},
        }
    )
    assignment = pd.DataFrame(
        {"id": X.index, "partition": ["dev"] * 32, "fold": [0] * 16 + [1] * 16}
    )
    result = cross_validate(
        build_adapter_factory(cfg),
        X,
        y,
        assignment,
        "id",
        validation_fraction=0.25,
        seed=2,
        metric_names=["roc_auc"],
        alert_rates=[],
    )
    assert len(result.oof) == 32
    for k, adapter in result.fold_models.items():
        path = tmp_path / f"fold_{k}.ubj"
        adapter.save(path)
        loaded = load_adapter(cfg, path)
        np.testing.assert_allclose(
            adapter.predict_proba(X), loaded.predict_proba(X), rtol=0, atol=1e-6
        )


def test_fitted_wrapper_composition(tmp_path):
    from ml_template.features.fitted_features import TransformingAdapter

    X, y = frame()
    X["x"] = np.arange(32, dtype=float) + 1
    specs = [{"step": "power_transform", "columns": ["x"], "method": "yeo-johnson"}]
    a = TransformingAdapter(NNAdapter(params(), 3), specs)
    a.refit(X, y, 1)
    path = tmp_path / "wrapped.ubj"
    a.save(path)
    b = TransformingAdapter.load(NNAdapter.load(params(), 3, path), specs, path)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=0, atol=1e-6)


def test_actual_training_smoke(tmp_path):
    from ml_template.training.nn_smoke import SmokeConfig, run_smoke

    # Independent of the user's editable model configuration.
    model_file = tmp_path / "nn_model.yaml"
    model_file.write_text(
        "type: nn\nparams: {}\n",
        encoding="utf-8",
    )

    report = run_smoke(
        SmokeConfig(
            model_file=str(model_file),
            sample_rows=32,
            overrides={
                "device": "cpu",
                "inference_device": "cpu",
                "max_epochs": 2,
                "patience": 1,
                "hidden_dims": [8],
                "batch_size": 7,
            },
        )
    )

    assert report["status"] == "passed"
    assert report["training_device"] == "cpu"
    assert report["inference_device"] == "cpu"
    assert report["holdout_labels_loaded"] is False
