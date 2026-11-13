import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.explainability.regression_result import (
    RegressionTolerance,
    build_regression_explanation,
    global_summary,
)


def sample():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan]}, index=pd.Index(["x", "y"], name="id"))
    phi = pd.DataFrame({"a": [2.0, -3.0], "b": [-1.0, 1.0]}, index=X.index)
    base = pd.Series([10.0, -5.0], index=X.index)
    return X, phi, base, pd.Series([11.0, -7.0], index=X.index)


def test_continuous_predictions_baseline_and_owned_copies():
    X, phi, base, predicted = sample()
    result = build_regression_explanation(X, phi, base, predicted)
    np.testing.assert_allclose(result.rows.y_pred_reconstructed, [11.0, -7.0])
    assert result.output_scale == "prediction_target_units"
    X.iloc[0, 0] = 999
    phi.iloc[0, 0] = 999
    assert result.features.a.iloc[0] == 1
    assert result.contributions.a.iloc[0] == 2
    json.dumps(result.verification, allow_nan=False)


@pytest.mark.parametrize(
    "fault", ["additivity", "shape", "nan", "boolean", "strings", "swapped_labels"]
)
def test_invalid_contributions_rejected(fault):
    X, phi, base, predicted = sample()
    if fault == "additivity":
        phi = phi + 1
    elif fault == "shape":
        phi = np.zeros((2, 3))
    elif fault == "nan":
        phi.iloc[0, 0] = np.nan
    elif fault == "boolean":
        phi = np.zeros((2, 2), dtype=bool)
    elif fault == "strings":
        phi = np.array([["2", "-1"], ["-3", "1"]])
    else:
        phi = phi.iloc[::-1]
    with pytest.raises(ValueError):
        build_regression_explanation(X, phi, base, predicted)


@pytest.mark.parametrize(
    "fault", ["duplicate_key", "null_key", "duplicate_feature", "multiindex", "empty"]
)
def test_invalid_features_rejected(fault):
    X, phi, base, predicted = sample()
    if fault == "duplicate_key":
        X.index = [1, 1]
    elif fault == "null_key":
        X.index = [1, None]
    elif fault == "duplicate_feature":
        X.columns = ["a", "a"]
    elif fault == "multiindex":
        X.index = pd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    else:
        X = X.iloc[:0]
    with pytest.raises(ValueError):
        build_regression_explanation(X, phi, base, predicted)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, True, "1e-5"])
def test_invalid_tolerances(bad):
    with pytest.raises(ValueError):
        RegressionTolerance(prediction_atol=bad)


def test_prediction_label_mismatch():
    X, phi, base, predicted = sample()
    with pytest.raises(ValueError):
        build_regression_explanation(X, phi, base, predicted.iloc[::-1])


def test_global_summary_uses_mean_absolute_signed_contributions():
    result = build_regression_explanation(*sample())
    ranking = global_summary(result).set_index("feature")
    assert ranking.loc["a", "mean_abs_contribution"] == 2.5
    assert ranking.loc["a", "mean_contribution"] == -0.5
    assert ranking.loc["b", "mean_abs_contribution"] == 1


def test_cv_weighted_shap_explains_mean_prediction():
    X, phi, base, predicted = sample()
    phi2, base2 = phi * 2, base + 3
    combined = 0.5 * phi + 0.5 * phi2
    baseline = 0.5 * base + 0.5 * base2
    mean_prediction = 0.5 * predicted + 0.5 * (base2 + phi2.sum(axis=1))
    result = build_regression_explanation(X, combined, baseline, mean_prediction)
    np.testing.assert_allclose(result.rows.y_pred_reconstructed, mean_prediction)


def test_regression_distribution_figure_uses_target_units():
    pytest.importorskip("plotly")
    from ml_template.explainability.regression_figures import distribution_figure

    result = build_regression_explanation(*sample())
    view = SimpleNamespace(
        rows=result.rows,
        features=result.features,
        contributions=result.contributions,
        importance=global_summary(result),
        summary={"target_unit": "target_units", "varying_feature_values": []},
    )
    figure = distribution_figure(view, max_points=1)
    assert "target_units" in figure.layout.xaxis.title.text
    assert len(figure.data) == 2
    assert len(figure.data[0].x) == 1
