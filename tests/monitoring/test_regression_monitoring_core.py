import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.monitoring.nullable_drift import nullable_compare, nullable_profile
from ml_template.monitoring.regression import (
    evaluate_predictions,
    findings,
    validate_options,
    validate_outcomes,
    validate_predictions,
)


def data():
    predictions = pd.DataFrame(
        {
            "id": [3, 1, 2, 4],
            "y_pred_raw": [7.0, -2.0, 3.0, 10.0],
            "y_pred_release": [7.0, -2.0, 3.0, 10.0],
        }
    )
    labels = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "y_true": [-1.5, 2.25, 8.5, None],
            "observed_at": ["2026-01-01T00:00:00Z"] * 3 + [None],
            "eligible_at": ["2026-01-01T00:00:00Z"] * 4,
        }
    )
    return predictions, labels


def evaluate(p, y, **kwargs):
    return evaluate_predictions(p, y, "id", as_of="2026-01-02T00:00:00Z", min_labelled=2, **kwargs)


def config(**overrides):
    values = dict(
        min_rows=2,
        min_labelled=2,
        min_label_coverage=0.9,
        js_warning=0.1,
        missing_rate_warning=0.2,
        missing_rate_delta_warning=0.1,
        target_unit="target_units",
        rmse_warning=0.1,
        mae_warning=None,
        abs_bias_warning=None,
        brier_warning=None,
        recall_floor=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def quality(n=4, missing=0):
    return {
        "n_rows": n,
        "features": [{"feature": "a", "n_rows": n, "missing": missing, "outside_range": 0}],
    }


def test_keyed_fractional_negative_outcomes_and_metric_signs():
    p, y = data()
    result = evaluate(p, y)
    metrics = result["score_metrics"]["y_pred_release"]
    residuals = np.array([0.5, -0.75, 1.5])
    assert result["n_evaluated"] == 3
    assert result["mature_label_coverage"] == 0.75
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(residuals**2)))
    assert metrics["mae"] == pytest.approx(np.mean(abs(residuals)))
    assert metrics["bias"] == pytest.approx(-residuals.mean())
    json.dumps(result, allow_nan=False)


def test_late_and_immature_outcomes_do_not_enter_metrics():
    p, y = data()
    y.loc[0, "observed_at"] = "2026-01-03T00:00:00Z"
    y.loc[1, "eligible_at"] = "2026-01-03T00:00:00Z"
    result = evaluate(p, y)
    assert result["n_evaluated"] == 1
    assert result["n_mature"] == 3
    assert not result["sufficient_support"]
    assert result["score_metrics"]["y_pred_release"]["r2"] is None


def test_all_missing_outcomes_remain_unknown():
    p, y = data()
    y["y_true"], y["observed_at"] = None, None
    result = evaluate(p, y)
    assert result["n_evaluated"] == 0 and result["score_metrics"] == {}
    assert result["mature_label_coverage"] == 0


def test_constant_targets_have_no_defined_r2():
    p, y = data()
    y.loc[y.y_true.notna(), "y_true"] = 1.5
    assert evaluate(p, y)["score_metrics"]["y_pred_release"]["r2"] is None


@pytest.mark.parametrize("bad", [np.inf, -np.inf, True, "3.14"])
def test_invalid_outcome_values(bad):
    with pytest.raises(ValueError):
        validate_outcomes(pd.Series([bad]))


@pytest.mark.parametrize(
    "fault", ["duplicate", "wrong_key", "missing_key", "bad_timestamp", "naive_time"]
)
def test_join_and_timestamp_validation(fault):
    p, y = data()
    if fault == "duplicate":
        y.loc[0, "id"] = 2
    elif fault == "wrong_key":
        y.loc[0, "id"] = 99
    elif fault == "missing_key":
        y.loc[0, "id"] = None
    elif fault == "bad_timestamp":
        y.loc[0, "observed_at"] = None
    else:
        y.loc[0, "eligible_at"] = "2026-01-01"
    with pytest.raises(ValueError):
        evaluate(p, y)


@pytest.mark.parametrize(
    "numeric,values",
    [
        (True, [1.0, 2.0, None, 4.0]),
        (False, ["a", "b", None, "a"]),
        (True, [None, None]),
        (False, [None, None]),
    ],
)
def test_nullable_self_reference_is_zero(numeric, values):
    series = pd.Series(values)
    frozen = nullable_profile(series, numeric=numeric)
    result = nullable_compare(series.iloc[::-1], frozen)
    assert result["js_divergence"] == pytest.approx(0)
    assert sum(result["current_counts"]) == len(series)
    json.dumps(result, allow_nan=False)


def test_current_all_missing_and_reference_all_missing():
    ref = nullable_profile(pd.Series([1.0, 2.0]), numeric=True)
    current = nullable_compare(pd.Series([None, None]), ref)
    assert current["js_divergence"] == pytest.approx(1)
    ref = nullable_profile(pd.Series([None, None]), numeric=True)
    current = nullable_compare(pd.Series([1.0, 2.0]), ref)
    assert current["observed_distribution_unavailable"]
    assert current["js_divergence"] == pytest.approx(1)


def test_missingness_is_distinct_from_real_category_text():
    ref = nullable_profile(pd.Series(["missing", None]), numeric=False)
    changed = nullable_compare(pd.Series(["new", None]), ref)
    assert changed["unseen_reference_categories"] == 1
    assert changed["missing_rate"] == 0.5


def test_out_of_range_values_are_counted_and_bins_frozen():
    ref = nullable_profile(pd.Series([1.0, 2.0, 3.0]), numeric=True)
    frozen = json.dumps(ref, sort_keys=True)
    result = nullable_compare(pd.Series([-100.0, 100.0, None]), ref)
    assert result["outside_reference_range"] == 2
    assert json.dumps(ref, sort_keys=True) == frozen
    assert sum(result["current_counts"]) == 3


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_nonfinite_drift_rejected(bad):
    with pytest.raises(ValueError):
        nullable_profile(pd.Series([1.0, bad]), numeric=True)
    ref = nullable_profile(pd.Series([1.0, 2.0]), numeric=True)
    with pytest.raises(ValueError):
        nullable_compare(pd.Series([bad]), ref)


def test_empty_drift_rejected():
    with pytest.raises(ValueError):
        nullable_profile(pd.Series([], dtype=float), numeric=True)


def test_no_labels_never_claims_predictive_health():
    result = findings(
        config(), drift=[], quality=quality(), performance=None, reference_n=4, sampling=None
    )
    assert any(x["rule"] == "no_labels" for x in result)


@pytest.mark.parametrize(
    "sampling,coverage,support",
    [
        ("selected", 1.0, True),
        ("unknown", 1.0, True),
        ("all_rows", 0.5, True),
        ("all_rows", 1.0, False),
    ],
)
def test_performance_warnings_require_representative_mature_support(sampling, coverage, support):
    result = {
        "mature_label_coverage": coverage,
        "sufficient_support": support,
        "n_evaluated": 3,
        "score_metrics": {"y_pred_release": {"rmse": 100}},
    }
    warnings = findings(
        config(), drift=[], quality=quality(), performance=result, reference_n=4, sampling=sampling
    )
    assert not any(x["rule"] == "rmse_warning" for x in warnings)


def test_supported_performance_and_missingness_warnings():
    performance = {
        "mature_label_coverage": 1.0,
        "sufficient_support": True,
        "n_evaluated": 4,
        "score_metrics": {"y_pred_release": {"rmse": 2.0, "bias": -4.0}},
    }
    measured = [{"name": "input:a", "js_divergence": 0.8, "missing_rate_delta": 0.3}]
    result = findings(
        config(abs_bias_warning=3),
        drift=measured,
        quality=quality(missing=2),
        performance=performance,
        reference_n=4,
        sampling="all_rows",
    )
    rules = {x["rule"] for x in result}
    assert {
        "missingness",
        "missingness_change",
        "distribution_change",
        "rmse_warning",
        "abs_bias_warning",
    } <= rules


@pytest.mark.parametrize("problem", ["unit", "reference_unit", "label_unit", "brier"])
def test_task_and_unit_binding(problem):
    cfg = config()
    reference = {"target_unit": "target_units"}
    labels = {"metadata": {"target_unit": "target_units"}}
    if problem == "unit":
        cfg.target_unit = None
    elif problem == "reference_unit":
        reference["target_unit"] = "other_units"
    elif problem == "label_unit":
        labels["metadata"]["target_unit"] = "other_units"
    else:
        cfg.brier_warning = 0.1
    with pytest.raises(ValueError):
        validate_options(
            SimpleNamespace(task="regression"), cfg, reference=reference, labels=labels
        )


def test_regression_predictions_are_not_probabilities():
    p, _ = data()
    info = {
        "task": "regression",
        "prediction_kind": "continuous",
        "release_variant": "raw",
        "calibration_id": None,
        "policy": {"kind": "none"},
        "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
    }
    summary = {**info, "n_flagged": None}
    validate_predictions(p, summary, info)
    p.loc[0, "y_pred_release"] = 0
    with pytest.raises(ValueError):
        validate_predictions(p, summary, info)


def test_probability_flags_cannot_enter_regression_monitoring():
    p, _ = data()
    info = {
        "task": "regression",
        "prediction_kind": "continuous",
        "release_variant": "raw",
        "calibration_id": None,
        "policy": {"kind": "none"},
        "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
    }
    with pytest.raises(ValueError):
        validate_predictions(p.assign(policy_flag=True), {**info, "n_flagged": None}, info)
