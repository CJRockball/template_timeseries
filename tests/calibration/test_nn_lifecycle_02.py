"""Torch-free calibration request, role boundary and shared mathematical tests."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.models.regression_postprocessing import (
    apply,
    correct,
    fit_correction,
    fit_quantiles,
    rank_for,
)
from ml_template.training.regression_calibration import (
    CalibrationConfig,
    _load_reserved_partition,
    calibration_roles,
    fingerprint,
    prediction_binding,
)


def config(**extra):
    return CalibrationConfig(
        **{
            "model_type": "nn",
            "parent_run_id": "run_test",
            "target_unit": "MPa",
            "intended_use": "Synthetic test",
            "min_calibration_rows": 20,
            "min_recalibration_rows": 3,
            **extra,
        }
    )


def prepared():
    a = pd.DataFrame(
        {
            "id": range(55),
            "partition": ["dev"] * 20
            + ["holdout"] * 5
            + ["calibration"] * 20
            + ["recalibration"] * 10,
        }
    )
    return SimpleNamespace(assignments=a, key_col="id")


def test_xgb_default_serialization_unchanged():
    cfg = CalibrationConfig(
        parent_run_id="run_test", target_unit="MPa", intended_use="Synthetic test"
    )
    expected = {
        "parent_run_id": "run_test",
        "predictor_kind": "final",
        "correction": "none",
        "coverages": [0.8, 0.9, 0.95],
        "target_unit": "MPa",
        "min_calibration_rows": 30,
        "min_recalibration_rows": 10,
        "intended_use": "Synthetic test",
        "acknowledge_prior_exposure": False,
        "prior_exposure_note": "",
    }
    assert cfg.model_type == "xgb"
    assert json.loads(cfg.model_dump_json()) == expected
    assert fingerprint(cfg.model_dump(mode="json")) == fingerprint(expected)


def test_nn_model_type_is_explicit_in_artifact_request():
    assert config().model_dump(mode="json")["model_type"] == "nn"
    assert CalibrationConfig.model_validate_json(config().model_dump_json()) == config()


@pytest.mark.parametrize(
    "extra",
    [
        {"model_type": "torch"},
        {"predictor_kind": "auto"},
        {"correction": "isotonic"},
        {"coverages": [0.9, 0.8]},
        {"coverages": [0.8, 0.8]},
        {"coverages": [1.0]},
        {"coverages": [-0.1]},
        {"coverages": [0.1234567]},
        {"coverages": [], "correction": "none"},
        {"target_unit": " "},
        {"intended_use": " "},
        {"acknowledge_prior_exposure": True, "prior_exposure_note": ""},
        {"calibration_partition": "dev"},
    ],
)
def test_invalid_requests(extra):
    with pytest.raises(ValueError):
        config(**extra)


@pytest.mark.parametrize("method", ["none", "bias", "affine"])
def test_only_required_roles_selected(method):
    p = prepared()
    roles = calibration_roles(p, config(correction=method))
    assert roles["calibration"] == list(range(25, 45))
    assert roles["recalibration"] == ([] if method == "none" else list(range(45, 55)))
    assert not (set(roles["calibration"]) | set(roles["recalibration"])) & set(range(25))


def test_correction_only_does_not_use_interval_calibration():
    roles = calibration_roles(prepared(), config(correction="bias", coverages=[]))
    assert roles["calibration"] == [] and roles["recalibration"] == list(range(45, 55))


@pytest.mark.parametrize("role", ["calibration", "recalibration"])
def test_missing_reserved_population_rejected(role):
    p = prepared()
    p.assignments.loc[p.assignments.partition.eq(role), "partition"] = "dev"
    with pytest.raises(ValueError, match="BEFORE training"):
        calibration_roles(p, config(correction="bias"))


def test_no_finite_interval_for_undersized_population():
    with pytest.raises(ValueError, match="finite requested interval"):
        calibration_roles(prepared(), config(coverages=[0.99]))


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unknown_role"])
def test_malformed_population(kind):
    p = prepared()
    if kind == "duplicate":
        p.assignments.loc[25, "id"] = 0
    elif kind == "missing":
        p.assignments.loc[25, "id"] = np.nan
    else:
        p.assignments.loc[25, "partition"] = "other"
    with pytest.raises(ValueError):
        calibration_roles(p, config(correction="bias"))


@pytest.mark.parametrize(
    "role,keys",
    [
        ("dev", [0]),
        ("holdout", [20]),
        ("other", [0]),
        ("calibration", []),
        ("calibration", [25, 25]),
        ("calibration", list(range(26, 45))),
        ("calibration", list(range(20))),
        ("recalibration", list(range(25, 45))),
    ],
)
def test_invalid_loads_blocked_before_partition_loader(role, keys):
    with pytest.raises(ValueError):
        _load_reserved_partition(prepared(), role, keys)


@pytest.mark.parametrize(
    "method,slope,intercept", [("none", 1.0, 0.0), ("bias", 1.0, 3.0), ("affine", 2.0, 3.0)]
)
def test_known_corrections(method, slope, intercept):
    p = np.array([-2.0, -0.5, 1.0, 3.0])
    y = p if method == "none" else (p + 3 if method == "bias" else 2 * p + 3)
    fitted = fit_correction(p, y, method)
    assert fitted["slope"] == pytest.approx(slope)
    assert fitted["intercept"] == pytest.approx(intercept)
    np.testing.assert_allclose(correct(p, fitted), y)


def test_constant_predictor_affine_rejected():
    with pytest.raises(ValueError, match="varying predictions"):
        fit_correction([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], "affine")


def test_bias_then_independent_quantile_and_apply():
    correction = fit_correction([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], "bias")
    prediction = np.zeros(9)
    target = np.arange(9.0) + 3
    quantiles = fit_quantiles(correct(prediction, correction), target, [0.5, 0.8])
    assert quantiles["c500000"]["rank"] == 5 and quantiles["c500000"]["half_width"] == 4
    assert quantiles["c800000"]["rank"] == 8 and quantiles["c800000"]["half_width"] == 7
    state = {
        "schema_version": 1,
        "correction": correction,
        "n_calibration": 9,
        "quantiles": quantiles,
    }
    point, bounds = apply([-1.0, 2.0], state, [0.8])
    np.testing.assert_allclose(point, [2.0, 5.0])
    np.testing.assert_allclose(bounds["pi_lower_c800000"], point - 7)
    np.testing.assert_allclose(bounds["pi_upper_c800000"], point + 7)


@pytest.mark.parametrize(
    "n,coverage,rank", [(19, 0.95, 19), (20, 0.95, 20), (8, 0.8, 8), (8, 0.95, 9)]
)
def test_finite_sample_rank(n, coverage, rank):
    assert rank_for(n, coverage) == rank


def provenance():
    return {
        "predictor": {"model_type": "nn", "kind": "final"},
        "input_hashes": {"model.ubj": "a"},
        "feature_preparation": {"feature_set_id": None},
        "versions": {"torch": "test"},
        "python": "test",
        "code_hashes": {
            "models/nn_regression_predictor.py": "a",
            "evaluation/regression_inputs.py": "b",
            "dashboard/app.py": "c",
        },
    }


@pytest.mark.parametrize(
    "field",
    ["predictor", "input_hashes", "feature_preparation", "versions", "python", "code_hashes"],
)
def test_binding_changes_with_critical_evidence(field):
    p = provenance()
    changed = copy.deepcopy(p)
    if field == "code_hashes":
        changed[field]["models/nn_regression_predictor.py"] = "new"
    else:
        changed[field] = {"changed": True} if field != "python" else "new"
    assert fingerprint(prediction_binding(changed)) != fingerprint(prediction_binding(p))


def test_dashboard_only_code_not_prediction_binding():
    p = provenance()
    changed = copy.deepcopy(p)
    changed["code_hashes"]["dashboard/app.py"] = "new"
    assert prediction_binding(changed) == prediction_binding(p)
