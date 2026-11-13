import pytest
from pydantic import ValidationError

from ml_template.training.calibration_config import CalibrationConfig, load_calibration_config


def test_test_parent_config(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text("parent_run_id: run_20261016_121745_958bec\nmethod: sigmoid\n")
    assert load_calibration_config(path).parent_run_id == "run_20261016_121745_958bec"


@pytest.mark.parametrize(
    "extra",
    [{"epsilon": float("nan")}, {"method": "platt"}, {"parent_run_id": "../bad"}, {"typo": 1}],
)
def test_config_rejects_invalid(extra):
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate({"parent_run_id": "run_test", **extra})
