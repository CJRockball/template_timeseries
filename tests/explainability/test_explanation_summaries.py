import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ml_template.explainability.result import build_margin_explanation
from ml_template.explainability.summaries import feature_summary, grouped_summary


def example():
    X = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    phi = np.array([[2.0, -2.0], [1.0, -1.0]])
    return build_margin_explanation(X, phi, [-1, -1], [-1, -1], expit([-1, -1]))


def test_feature_summary():
    out = feature_summary(example())
    assert out.feature.tolist() == ["x", "y"]
    assert out.mean_abs_contribution.tolist() == [1.5, 1.5]
    assert out.n_rows.tolist() == [2, 2]
    assert out.positive_fraction.tolist() == [1, 0]


def test_groups_sum_signed_before_absolute():
    out = grouped_summary(example(), {"combined": ["x", "y"]})
    assert out.mean_abs_contribution.iloc[0] == 0
    assert out.n_features.iloc[0] == 2


@pytest.mark.parametrize(
    "groups",
    [
        {},
        {"a": ["x"]},
        {"a": ["x", "y"], "b": ["y"]},
        {"a": ["x", "missing"]},
        {"a": "xy"},
        {"a": []},
        {1: ["x", "y"]},
        {"a": [1, 2]},
    ],
)
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        grouped_summary(example(), groups)
