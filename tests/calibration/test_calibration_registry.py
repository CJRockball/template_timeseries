import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from ml_template.db.calibration_registry import (  # noqa: E402
    calibration_metrics,
    finish_calibration,
    get_calibration,
    start_calibration,
)


def test_success_and_separate_metrics():
    engine = sqlalchemy.create_engine("sqlite://")
    start_calibration(engine, "cal_test", "run_parent", "sigmoid", {}, "artifacts/cal_test")
    finish_calibration(
        engine,
        "cal_test",
        succeeded=True,
        rows=[
            {
                "split": "dev_calibration_diagnostic",
                "variant": "raw",
                "fold": -1,
                "metric": "brier",
                "value": 0.2,
            }
        ],
    )
    assert get_calibration(engine, "cal_test")["status"] == "succeeded"
    with engine.connect() as conn:
        assert conn.execute(sqlalchemy.select(calibration_metrics.c.value)).scalar_one() == 0.2
    assert "run_metrics" not in sqlalchemy.inspect(engine).get_table_names()


def test_failure_record():
    engine = sqlalchemy.create_engine("sqlite://")
    start_calibration(engine, "cal_test", "run_parent", "none", {}, "artifacts/cal_test")
    finish_calibration(engine, "cal_test", succeeded=False, error="test failure")
    assert get_calibration(engine, "cal_test")["status"] == "failed"
