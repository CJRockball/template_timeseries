import pytest

from ml_template.training.regression_calibration import CalibrationConfig


def config(**kwargs):
    return CalibrationConfig(
        parent_run_id="run_test", target_unit="units", intended_use="Synthetic test", **kwargs
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(coverages=[0.9, 0.8]),
        dict(correction="isotonic"),
        dict(predictor_kind="choose_best"),
        dict(coverages=[], correction="none"),
        dict(acknowledge_prior_exposure=True),
    ],
)
def test_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_point_only_and_intervals_are_explicit():
    assert config(correction="bias", coverages=[]).coverages == ()
    assert config(correction="none", coverages=[0.9]).coverages == (0.9,)
