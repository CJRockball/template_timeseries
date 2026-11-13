import pytest

from ml_template.dashboard.threshold_figs import fold_optima, objective_overlay, operating_curves
from ml_template.evaluation.threshold import fold_recommendations, threshold_profile


def test_overlay_marks_optimum_and_current_cutoff():
    profile = threshold_profile([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
    fig = objective_overlay({"run": profile}, "f1", 0.5)
    assert len(fig.data) == 2 and fig.data[1].marker.symbol == "diamond"
    assert fig.data[1].x[0] == 0.8 and len(fig.layout.shapes) == 1
    assert fig.data[0].line.shape == "vh"


def test_operating_curves_and_fold_plot():
    y, p = [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]
    assert len(operating_curves(threshold_profile(y, p)).data) == 3
    fig = fold_optima(fold_recommendations(y, p, [0, 0, 1, 1]))
    assert len(fig.data) == 2


def test_undefined_and_empty_overlays():
    assert objective_overlay({}).layout.annotations
    profile = threshold_profile([0, 0], [0.1, 0.9])
    assert not objective_overlay({"run": profile}).data
    with pytest.raises(ValueError):
        objective_overlay({}, "roc_auc")
