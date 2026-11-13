import numpy as np
import pytest

from ml_template.evaluation.calibration_diagnostics import (
    cross_fitted_probabilities,
    probability_metrics,
    reliability_table,
)
from ml_template.models.probability_calibrator import ProbabilityCalibrator


def fixture():
    rng = np.random.default_rng(45)
    p = rng.uniform(0.01, 0.99, 600)
    y = rng.binomial(1, p * 0.4)
    return p, y, np.arange(len(p)) % 3


def test_crossfit_matches_explicit_exclusion():
    p, y, folds = fixture()
    output = cross_fitted_probabilities(p, y, folds, min_class_count=2)
    for fold in range(3):
        mask = folds == fold
        cal = ProbabilityCalibrator.fit(p[~mask], y[~mask])
        assert np.allclose(output[mask], cal.transform(p[mask]))


def test_held_fold_labels_cannot_change_its_predictions():
    p, y, folds = fixture()
    original = cross_fitted_probabilities(p, y, folds, min_class_count=2)
    changed_y = y.copy()
    changed_y[folds == 0] = 1 - changed_y[folds == 0]
    changed = cross_fitted_probabilities(p, changed_y, folds, min_class_count=2)
    assert np.array_equal(original[folds == 0], changed[folds == 0])


def test_reliability_retains_support_and_ties():
    y = [0, 0, 0, 1, 1, 1]
    table = reliability_table(y, [0.2] * 6)
    assert len(table) == 1
    assert table.n_rows.sum() == 6
    assert table.n_positives.sum() == 3
    assert table.event_rate_low.iloc[0] < 0.5 < table.event_rate_high.iloc[0]


def test_probability_metrics_one_class():
    metrics = probability_metrics([0, 0], [0.1, 0.2])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["log_loss"] > 0


def test_insufficient_class_counts():
    with pytest.raises(ValueError, match="too few"):
        cross_fitted_probabilities([0.2] * 8, [0] * 7 + [1], [0, 1] * 4)
