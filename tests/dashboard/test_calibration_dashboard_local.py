"""Optional read-only smoke test against the user's actual saved calibration."""

import os

import pytest

pytestmark = [
    pytest.mark.local_data,
    pytest.mark.skipif(
        os.environ.get("ML_TEMPLATE_CALIBRATION_DASHBOARD_LOCAL") != "1",
        reason="requires the saved local calibration artifacts",
    ),
]


def test_saved_calibration_260e14():
    from ml_template.dashboard.calibration_data import load_calibration, metric_comparison
    from ml_template.db.connection import get_engine

    bundle = load_calibration(get_engine(), "cal_20261017_063551_260e14")
    assert bundle["header"]["parent_run_id"] == "run_20261016_121745_958bec"
    assert bundle["provenance"]["feature_set_id"] == "fs_d0383e096267"
    assert len(bundle["predictions"]) == 8000
    assert bundle["predictions"]["y_true"].sum() == 271
    assert {"brier", "log_loss"}.issubset(set(metric_comparison(bundle["metrics"]).metric))
    assert bundle["inventory"].state.eq("verified").all()
