import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ml_template.explainability.result import ExplanationTolerance, build_margin_explanation


def inputs():
    X = pd.DataFrame(
        {"x": [1.0, 2.0], "kind": pd.Categorical(["L", "H"])},
        index=pd.Index([20, 10], name="entity"),
    )
    phi = np.array([[0.1, -0.2], [0.3, 0.4]])
    base = np.array([-1.0, -1.0])
    margin = base + phi.sum(axis=1)
    return [X, phi, base, margin, expit(margin)]


def test_result_preserves_keys_categories_and_copies():
    args = inputs()
    out = build_margin_explanation(*args)
    assert out.features.index.tolist() == [20, 10]
    assert isinstance(out.features.kind.dtype, pd.CategoricalDtype)
    assert out.verification["passed"]
    assert out.output_scale == "base_model_log_odds"
    args[0].iloc[0, 0] = 100
    args[1][0, 0] = 100
    assert out.features.x.iloc[0] == 1
    assert out.contributions.x.iloc[0] == 0.1


@pytest.mark.parametrize("position", [1, 2, 3, 4])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_rejected(position, value):
    args = inputs()
    args[position].flat[0] = value
    with pytest.raises(ValueError):
        build_margin_explanation(*args)


@pytest.mark.parametrize("position", [1, 2, 3, 4])
def test_shape_rejected(position):
    args = inputs()
    args[position] = args[position][:1]
    with pytest.raises(ValueError):
        build_margin_explanation(*args)


def test_additivity_failure():
    args = inputs()
    args[1][0, 0] += 1
    with pytest.raises(ValueError, match="reconstruct"):
        build_margin_explanation(*args)


def test_probability_failure():
    args = inputs()
    args[4][0] = 0.99
    with pytest.raises(ValueError, match="probability"):
        build_margin_explanation(*args)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_probability_bounds(value):
    args = inputs()
    args[4][0] = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_margin_explanation(*args)


@pytest.mark.parametrize("index", [[1, 1], [1, None]])
def test_bad_keys(index):
    args = inputs()
    args[0].index = index
    with pytest.raises(ValueError, match="keys"):
        build_margin_explanation(*args)


def test_duplicate_columns():
    args = inputs()
    args[0].columns = ["same", "same"]
    with pytest.raises(ValueError, match="feature names"):
        build_margin_explanation(*args)


def test_empty_frame():
    args = inputs()
    args[0] = args[0].iloc[:0]
    with pytest.raises(ValueError, match="nonempty"):
        build_margin_explanation(*args)


def test_dataframe_contribution_alignment():
    args = inputs()
    args[1] = pd.DataFrame(args[1], index=[10, 20], columns=args[0].columns)
    with pytest.raises(ValueError, match="labels"):
        build_margin_explanation(*args)


def test_series_alignment():
    args = inputs()
    args[2] = pd.Series(args[2], index=[10, 20])
    with pytest.raises(ValueError, match="labels"):
        build_margin_explanation(*args)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "1"])
def test_bad_tolerance(value):
    with pytest.raises(ValueError, match="tolerances"):
        ExplanationTolerance(margin_atol=value)


@pytest.mark.parametrize("value", [{}, {"margin_atol": 1}])
def test_wrong_tolerance_type(value):
    with pytest.raises(TypeError):
        build_margin_explanation(*inputs(), tolerance=value)


def test_exact_tolerance():
    out = build_margin_explanation(*inputs(), tolerance=ExplanationTolerance(0, 0, 0, 0))
    assert out.verification["max_abs_margin_residual"] == 0
