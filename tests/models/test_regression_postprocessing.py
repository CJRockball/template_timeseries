import copy
import json

import numpy as np
import pytest

from ml_template.models.regression_postprocessing import (
    apply,
    correct,
    coverage_tag,
    fit_correction,
    fit_quantiles,
    interval_columns,
    interval_metrics,
    levels,
    rank_for,
    validate_parameters,
)


def state(method="none"):
    return {
        "schema_version": 1,
        "correction": {
            "method": method,
            "slope": 1.0,
            "intercept": 2.0 if method == "bias" else 0.0,
        },
        "n_calibration": 99,
        "quantiles": fit_quantiles(np.zeros(99), np.arange(99.0), [0.8, 0.9, 0.95]),
    }


def test_exact_integer_ranks_not_interpolation():
    q = fit_quantiles(np.zeros(19), np.arange(19.0), [0.9, 0.95])
    assert q["c900000"] == {"coverage": 0.9, "rank": 18, "half_width": 17.0}
    assert q["c950000"]["rank"] == 19
    assert rank_for(99, 0.58) == 58
    assert rank_for(29, 0.7) == 21


@pytest.mark.parametrize("coverage", [0.9, 0.95, 0.99])
def test_small_sample_refuses_infinite_state(coverage):
    with pytest.raises(ValueError):
        fit_quantiles(np.zeros(2), np.zeros(2), [coverage])


@pytest.mark.parametrize("bad", [0, 1, -1, np.nan, np.inf, True, 0.1234567])
def test_invalid_coverage(bad):
    with pytest.raises(ValueError):
        coverage_tag(bad)


@pytest.mark.parametrize("bad", [[0.9, 0.8], [0.9, 0.9], "0.9"])
def test_explicit_ordered_levels(bad):
    with pytest.raises(ValueError):
        levels(bad)


def test_bias_affine_and_constant_prediction_rejection():
    p = np.arange(-5.0, 5.0)
    bias = fit_correction(p, p + 3, "bias")
    affine = fit_correction(p, 2 * p - 7, "affine")
    np.testing.assert_allclose(correct(p, bias), p + 3)
    np.testing.assert_allclose(correct(p, affine), 2 * p - 7)
    with pytest.raises(ValueError):
        fit_correction(np.ones(10), p, "affine")


def test_quantile_fit_cannot_mutate_correction():
    c = fit_correction(np.arange(10.0), np.arange(10.0) + 2, "bias")
    before = dict(c)
    fit_quantiles(correct(np.arange(99.0), c), np.arange(99.0) + 100, [0.9])
    assert c == before


def test_identity_nested_bounds_and_json_roundtrip():
    s = state()
    p = np.array([-100.0, 0.5, 200.0])
    point, bounds = apply(p, s, [0.8, 0.9, 0.95])
    np.testing.assert_array_equal(p, point)
    assert np.all(bounds["pi_lower_c950000"] <= bounds["pi_lower_c800000"])
    assert np.all(bounds["pi_upper_c950000"] >= bounds["pi_upper_c800000"])
    assert validate_parameters(json.loads(json.dumps(s, allow_nan=False))) == s


def test_corrected_interval_center():
    point, bounds = apply(np.array([-2.0, 3.0]), state("bias"), [0.9])
    np.testing.assert_allclose(point, [0.0, 5.0])
    np.testing.assert_allclose((bounds["pi_lower_c900000"] + bounds["pi_upper_c900000"]) / 2, point)


@pytest.mark.parametrize(
    "fault", ["rank", "negative", "nan", "extra", "slope", "identity", "unnested"]
)
def test_tampered_parameters(fault):
    s = copy.deepcopy(state())
    if fault == "rank":
        s["quantiles"]["c900000"]["rank"] = 1
    elif fault == "negative":
        s["quantiles"]["c900000"]["half_width"] = -1
    elif fault == "nan":
        s["correction"]["intercept"] = np.nan
    elif fault == "extra":
        s["extra"] = 1
    elif fault == "slope":
        s["correction"]["slope"] = True
    elif fault == "identity":
        s["correction"]["intercept"] = 1
    else:
        s["quantiles"]["c800000"]["half_width"] = 1000
    with pytest.raises(ValueError):
        validate_parameters(s)


def test_unfitted_level_rejected_and_batch_order_parity():
    s = state()
    with pytest.raises(ValueError):
        apply(np.zeros(3), s, [0.85])
    p = np.array([1.0, 2.0, -3.0, 5.0])
    whole, bounds = apply(p, s, [0.9])
    reverse, reverse_bounds = apply(p[::-1], s, [0.9])
    np.testing.assert_array_equal(whole, reverse[::-1])
    for name, values in bounds.items():
        np.testing.assert_array_equal(values, reverse_bounds[name][::-1])
        np.testing.assert_array_equal(
            values, [apply(p[i : i + 1], s, [0.9])[1][name][0] for i in range(len(p))]
        )


def test_empirical_coverage_inclusive_width_and_inversion():
    s = state()
    _, bounds = apply(np.array([0.0, 0.0]), s, [0.9])
    definitions = interval_columns(s, [0.9])
    y = np.array([bounds["pi_lower_c900000"][0], 100000.0])
    result = interval_metrics(y, bounds, definitions)["c900000"]
    assert result["empirical_coverage"] == 0.5
    assert result["mean_width"] == 2 * s["quantiles"]["c900000"]["half_width"]
    with pytest.raises(ValueError):
        interval_metrics(
            y, {"pi_lower_c900000": [2.0, 2.0], "pi_upper_c900000": [1.0, 1.0]}, definitions
        )


@pytest.mark.parametrize("bad", [[np.inf], [np.nan], [True], ["1"]])
def test_invalid_predictions(bad):
    with pytest.raises(ValueError):
        apply(bad, state(), [0.9])


@pytest.mark.parametrize("slope,intercept", [(1.0, 0.0), (1.0, 3.0), (-2.0, 7.0), (0.0, 5.0)])
def test_affine_shap_linearity(slope, intercept):
    phi = np.array([[1.0, -2.0], [-3.0, 4.0]])
    baseline = np.array([10.0, 10.0])
    raw = baseline + phi.sum(axis=1)
    np.testing.assert_allclose(
        baseline * slope + intercept + (phi * slope).sum(axis=1), raw * slope + intercept
    )
