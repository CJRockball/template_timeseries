"""Synthetic orchestration tests; require the normal project dependencies."""

import numpy as np
import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("pyarrow")


def setup_inputs(monkeypatch):
    from ml_template.training import calibrate
    from ml_template.training.calibration_inputs import CalibrationInputs

    rng = np.random.default_rng(51)
    n = 600
    p = rng.uniform(0.01, 0.99, n)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "fold": np.arange(n) % 3,
            "p_raw": p,
            "y_true": rng.binomial(1, p * 0.4),
        }
    )
    inputs = CalibrationInputs(
        frame, "id", {"parent_holdout_evaluated_at": None, "parent_run_id": "run_test"}
    )
    monkeypatch.setattr(calibrate, "load_calibration_inputs", lambda *a, **k: inputs)
    return calibrate, inputs


def test_synthetic_workflow(monkeypatch, tmp_path):
    from ml_template.db.calibration_registry import get_calibration
    from ml_template.models.probability_calibrator import ProbabilityCalibrator
    from ml_template.tracking.run_artifacts import verify_artifact
    from ml_template.training.calibration_config import CalibrationConfig

    calibrate, inputs = setup_inputs(monkeypatch)
    engine = sqlalchemy.create_engine("sqlite://")
    cfg = CalibrationConfig(parent_run_id="run_test")
    result = calibrate.calibrate_run(engine, cfg, output_root=tmp_path)
    out = tmp_path / result["calibration_id"]
    assert get_calibration(engine, result["calibration_id"])["status"] == "succeeded"
    for name in ["calibrator.json", "development_predictions.parquet", "reliability.csv"]:
        verify_artifact(out, name)
    saved = pd.read_parquet(out / "development_predictions.parquet")
    cal = ProbabilityCalibrator.load(out / "calibrator.json")
    assert np.array_equal(cal.transform(inputs.frame.p_raw), saved.p_cal_fit)
    assert not list(out.glob("*holdout*"))
    assert "model_runs" not in sqlalchemy.inspect(engine).get_table_names()


def test_check_does_not_create_calibration_records(monkeypatch, tmp_path):
    from ml_template.training.calibration_config import CalibrationConfig

    calibrate, _ = setup_inputs(monkeypatch)
    engine = sqlalchemy.create_engine("sqlite://")
    result = calibrate.calibrate_run(
        engine, CalibrationConfig(parent_run_id="run_test"), output_root=tmp_path, check_only=True
    )
    assert result["n_rows"] == 600
    assert "calibration_runs" not in sqlalchemy.inspect(engine).get_table_names()
    assert not list(tmp_path.iterdir())


def test_failure_is_recorded(monkeypatch, tmp_path):
    from ml_template.db.calibration_registry import calibration_runs
    from ml_template.training.calibration_config import CalibrationConfig

    calibrate, inputs = setup_inputs(monkeypatch)
    inputs.frame["fold"] = 0
    engine = sqlalchemy.create_engine("sqlite://")
    with pytest.raises(ValueError, match="at least two"):
        calibrate.calibrate_run(
            engine, CalibrationConfig(parent_run_id="run_test"), output_root=tmp_path
        )
    with engine.connect() as conn:
        status = conn.execute(sqlalchemy.select(calibration_runs.c.status)).scalar_one()
    assert status == "failed"
