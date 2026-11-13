"""Explicit opt-in: these tests use this checkout's real local data."""

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.local_data,
    pytest.mark.skipif(
        os.environ.get("ML_TEMPLATE_CALIBRATION_LOCAL") != "1",
        reason="set ML_TEMPLATE_CALIBRATION_LOCAL=1 to use local parent artifacts",
    ),
]


def test_parent_958bec_loads():
    from ml_template.db.connection import get_engine
    from ml_template.training.calibration_inputs import load_calibration_inputs

    inputs = load_calibration_inputs(get_engine(), "run_20261016_121745_958bec")
    assert len(inputs.frame) > 0
    assert inputs.frame[inputs.key_col].is_unique
    assert inputs.frame["y_true"].isin([0, 1]).all()
    assert inputs.provenance["parent_run_id"] == "run_20261016_121745_958bec"


def test_parent_958bec_calibration_roundtrip(tmp_path):
    from ml_template.db.connection import get_engine
    from ml_template.models.probability_calibrator import ProbabilityCalibrator
    from ml_template.training.calibration_inputs import load_calibration_inputs

    inputs = load_calibration_inputs(get_engine(), "run_20261016_121745_958bec")
    p, y = inputs.frame.p_raw.to_numpy(), inputs.frame.y_true.to_numpy()
    cal = ProbabilityCalibrator.fit(p, y)
    path = tmp_path / "calibrator.json"
    cal.save(path)
    assert np.array_equal(cal.transform(p), ProbabilityCalibrator.load(path).transform(p))
