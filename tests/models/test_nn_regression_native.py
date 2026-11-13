"""Real CPU NN adapter checks; must execute in the nn extra environment."""

import copy

import numpy as np
import pandas as pd
import pytest

from ml_template.models.nn.params import NNRegressionParams
from ml_template.models.nn_regressor import NNRegressionAdapter

pytest.importorskip("torch")


def params(**overrides):
    return NNRegressionParams(
        hidden_dims=(8,),
        max_epochs=3,
        patience=2,
        batch_size=4,
        dropout=0.0,
        embedding_dropout=0.0,
        prediction_batch_size=4,
        **overrides,
    )


def sample():
    X = pd.DataFrame(
        {"x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "cat": ["a", "b"] * 4},
        index=pd.Index(range(8), name="id"),
    )
    y = pd.Series(2 * X.x.to_numpy() - 3.25, index=X.index)
    return X, y


def test_training_checkpoint_and_keyed_chunk_parity(tmp_path):
    X, y = sample()
    adapter = NNRegressionAdapter(params(), 42)
    best = adapter.fit_with_validation(X.iloc[:6], y.iloc[:6], X.iloc[6:], y.iloc[6:])
    assert 1 <= best <= 3
    assert set(adapter.early_stopping_history.dataset) == {"inner_fit", "inner_stop"}
    assert adapter.early_stopping_history.selected_round.eq(best).all()
    assert not hasattr(adapter, "predict_proba")
    adapter.refit(X, y, best)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    with pytest.raises(FileExistsError):
        adapter.save(path)
    restored = NNRegressionAdapter.load(adapter.params, 42, path)
    expected = pd.Series(adapter.predict(X), index=X.index)
    reversed_singletons = pd.Series(
        [restored.predict(X.loc[[k]])[0] for k in X.index[::-1]], index=X.index[::-1]
    )
    np.testing.assert_allclose(
        expected.sort_index(), reversed_singletons.sort_index(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(restored.predict(X), expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        NNRegressionAdapter.load(adapter.params, 43, path)


def test_preprocessing_fits_training_not_validation_and_freezes_at_prediction():
    X, y = sample()
    validation = X.iloc[6:].copy()
    validation["x"] = 100000.0
    adapter = NNRegressionAdapter(params(), 42)
    adapter.fit_with_validation(X.iloc[:6], y.iloc[:6], validation, y.iloc[6:])
    state = adapter.preprocessor.state_dict()
    assert state["fit_rows"] == 6
    assert state["numeric_state"]["x"]["center"] == pytest.approx(X.x.iloc[:6].mean())
    changed = validation.copy()
    changed["x"] = np.nan
    changed["cat"] = "not-seen"
    arrays = adapter.preprocessor.transform_arrays(changed)
    assert np.isfinite(arrays.numeric).all() and (arrays.categorical == 1).all()
    assert (arrays.numeric[:, arrays.numeric_names.index("x__is_missing")] == 1).all()
    assert np.isfinite(adapter.predict(changed)).all()
    assert adapter.preprocessor.state_dict() == state
    with pytest.raises(ValueError):
        adapter.predict(changed.drop(columns="x"))


@pytest.mark.parametrize("branch", ["numeric", "categorical", "mixed"])
def test_input_branches_and_training_missing_values(branch):
    X, y = sample()
    X.loc[0, "x"] = np.nan
    X.loc[1, "cat"] = None
    X = X[["x"]] if branch == "numeric" else X[["cat"]] if branch == "categorical" else X
    adapter = NNRegressionAdapter(params(), 42)
    adapter.refit(X, y, 2)
    assert np.isfinite(adapter.predict(X)).all()


def test_real_outputs_are_not_sigmoid_clipped():
    import torch

    X, y = sample()
    adapter = NNRegressionAdapter(params(), 42)
    adapter.refit(X, y, 1)
    with torch.no_grad():
        for value in adapter.model.parameters():
            value.zero_()
        adapter.model.head[-1].bias.fill_(-2.5)
    np.testing.assert_allclose(adapter.predict(X), -2.5, rtol=0, atol=0)
    with torch.no_grad():
        adapter.model.head[-1].bias.fill_(4.25)
    np.testing.assert_allclose(adapter.predict(X), 4.25, rtol=0, atol=0)


def test_learning_curve_does_not_refit_original_model():
    X, y = sample()
    adapter = NNRegressionAdapter(params(), 42)
    adapter.refit(X, y, 1)
    before = adapter.predict(X)
    state = copy.deepcopy(adapter.preprocessor.state_dict())
    curves = adapter.learning_curve(X.iloc[:6], y.iloc[:6], X.iloc[6:], y.iloc[6:], 2, "mae")
    assert set(curves.dataset) == {"train", "val"} and set(curves.metric) == {"mae"}
    np.testing.assert_array_equal(before, adapter.predict(X))
    assert state == adapter.preprocessor.state_dict()


def test_checkpoint_task_and_nonfinite_weights_rejected(tmp_path):
    import torch

    X, y = sample()
    adapter = NNRegressionAdapter(params(), 42)
    adapter.refit(X, y, 1)
    path = tmp_path / "model.ubj"
    adapter.save(path)
    raw = torch.load(path, weights_only=True)
    raw["task"] = "binary_classification"
    bad = tmp_path / "bad_task.ubj"
    torch.save(raw, bad)
    with pytest.raises(ValueError):
        NNRegressionAdapter.load(adapter.params, 42, bad)
    raw["task"] = "regression"
    next(iter(raw["state_dict"].values())).fill_(float("nan"))
    bad = tmp_path / "bad_weights.ubj"
    torch.save(raw, bad)
    with pytest.raises(ValueError):
        NNRegressionAdapter.load(adapter.params, 42, bad)


def test_all_missing_numeric_and_categorical_inputs_are_finite():
    X, y = sample()
    X["x"] = np.nan
    X["cat"] = None
    adapter = NNRegressionAdapter(params(), 42)
    adapter.refit(X, y, 1)
    state = adapter.preprocessor.state_dict()
    assert state["numeric_state"]["x"]["n_observed"] == 0
    assert np.isfinite(adapter.predict(X)).all()
