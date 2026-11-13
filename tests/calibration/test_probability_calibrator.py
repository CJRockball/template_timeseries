import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import log_loss

from ml_template.models.probability_calibrator import ProbabilityCalibrator


def sample():
    rng = np.random.default_rng(34)
    z = rng.normal(size=4000)
    y = rng.binomial(1, expit(0.8 * z - 1.5))
    return expit(2 * z + 1), y


def test_sigmoid_reduces_synthetic_in_sample_loss():
    p, y = sample()
    cal = ProbabilityCalibrator.fit(p, y)
    assert log_loss(y, cal.transform(p)) < log_loss(y, p)
    assert cal.slope > 0


def test_order_and_roundtrip(tmp_path):
    p, y = sample()
    cal = ProbabilityCalibrator.fit(p, y)
    ordered = np.linspace(0, 1, 1000)
    assert np.all(np.diff(cal.transform(ordered)) >= 0)
    path = tmp_path / "calibrator.json"
    cal.save(path)
    assert np.array_equal(cal.transform(p), ProbabilityCalibrator.load(path).transform(p))


def test_none_is_exact_identity():
    p = np.array([0.0, 0.2, 0.6, 1.0])
    cal = ProbabilityCalibrator.fit(p, [0, 0, 1, 1], method="none")
    assert np.array_equal(p, cal.transform(p))


@pytest.mark.parametrize("p", [[np.nan, 0.5], [np.inf, 0.2], [-0.1, 0.5], [0.2, 1.1], [], [[0.1]]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        ProbabilityCalibrator().transform(p)


def test_invalid_labels():
    for y in [[0, 2], [0], [0, np.nan], [0, 0]]:
        with pytest.raises(ValueError):
            ProbabilityCalibrator.fit([0.2, 0.8], y)


def test_boundary_probabilities_are_finite():
    cal = ProbabilityCalibrator.fit([0, 0.1, 0.9, 1], [0, 0, 1, 1])
    assert np.isfinite(cal.transform([0, 1])).all()


def test_constant_scores():
    cal = ProbabilityCalibrator.fit(np.full(100, 0.5), [0] * 80 + [1] * 20)
    assert np.allclose(cal.transform([0.5]), [0.2], atol=1e-4)
