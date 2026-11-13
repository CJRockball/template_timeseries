import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import calibration_figs as cf


@pytest.mark.parametrize(
    "function",
    [cf.reliability_fig, cf.fold_metric_fig, cf.loss_fig, cf.distribution_fig, cf.mapping_fig],
)
def test_empty_states(function):
    figure = function(None)
    assert len(figure.layout.annotations) >= 1
    figure.to_json()


def test_reliability_error_bars_and_counts():
    table = pd.DataFrame(
        {
            "variant": ["raw", "calibrated_crossfit"],
            "mean_probability": [0.6, 0.3],
            "event_rate": [0.3, 0.3],
            "event_rate_low": [0.2, 0.2],
            "event_rate_high": [0.4, 0.4],
            "n_rows": [100, 100],
            "n_positives": [30, 30],
        }
    )
    figure = cf.reliability_fig(table)
    assert len(figure.data) == 3
    assert np.allclose(figure.data[1].error_y.array, [0.1])
    assert np.array_equal(figure.data[1].customdata, [[100, 30]])
    figure.to_json()


def test_fold_fig_and_missing_metric():
    metrics = pd.DataFrame(
        {
            "variant": ["raw", "calibrated_crossfit"],
            "fold": [0, 0],
            "metric": ["brier", "brier"],
            "value": [0.2, 0.1],
        }
    )
    assert len(cf.fold_metric_fig(metrics).data) == 2
    assert len(cf.fold_metric_fig(metrics, "roc_auc").layout.annotations) == 1


def test_loss_uses_separate_scales():
    table = pd.DataFrame(
        {"metric": ["brier", "log_loss"], "raw": [0.2, 0.6], "calibrated_crossfit": [0.1, 0.3]}
    )
    figure = cf.loss_fig(table)
    assert len(figure.data) == 4
    assert figure.data[2].xaxis == "x2"
    figure.to_json()


def test_final_mapping_is_monotone_and_not_identity():
    figure = cf.mapping_fig({"method": "sigmoid", "epsilon": 1e-7, "slope": 0.8, "intercept": -1})
    assert len(figure.data) == 2
    assert np.all(np.diff(figure.data[1].y) >= 0)
    assert not np.allclose(figure.data[1].x, figure.data[1].y)
    figure.to_json()


def test_none_mapping_is_identity():
    figure = cf.mapping_fig({"method": "none"})
    assert np.array_equal(figure.data[1].x, figure.data[1].y)


def test_invalid_final_mapping_is_unavailable():
    figure = cf.mapping_fig({"method": "sigmoid", "epsilon": 1e-7, "slope": -1, "intercept": 0})
    assert len(figure.data) == 0
    assert "invalid" in figure.layout.annotations[0].text


def test_distribution_aggregated_counts():
    table = pd.DataFrame(
        [{"variant": v, "class": c, "bin_mid": 0.5, "count": 20} for v in cf.COLORS for c in [0, 1]]
    )
    figure = cf.distribution_fig(table)
    assert len(figure.data) == 4
    figure.to_json()
