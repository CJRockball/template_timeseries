"""Continuous operational outcomes, maturity and descriptive warning rules.

Consumes saved predictions only. Never loads a model, fits or accesses holdout data.
Bias is mean(y_pred - y_true); residual quantiles use y_true - y_pred.
"""

import numpy as np
import pandas as pd

SCORE_COLUMNS = ("y_pred_raw", "y_pred_release")


def numeric(series, label):
    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError(label + " must be real numeric values")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(label + " must be finite")
    return values


def validate_predictions(frame, summary, info):
    if info.get("postprocessing") is not None:
        from ml_template.evaluation.regression_postprocessing import validate_saved_predictions

        return validate_saved_predictions(frame, summary, info)
    if (
        info.get("task") != "regression"
        or info.get("prediction_kind") != "continuous"
        or info.get("release_variant") != "raw"
        or info.get("calibration_id") is not None
        or info.get("policy", {}).get("kind") != "none"
    ):
        raise ValueError("unsupported regression release semantics")
    expected = {"raw": "y_pred_raw", "release": "y_pred_release"}
    if (
        summary.get("prediction_kind") != "continuous"
        or summary.get("prediction_columns") != expected
        or info.get("prediction_columns") != expected
        or summary.get("release_variant") != "raw"
        or summary.get("calibration_id") is not None
        or summary.get("n_flagged") is not None
    ):
        raise ValueError("regression summary semantics differ from release")
    if {"p_raw", "p_release", "p_calibrated", "policy_flag"} & set(frame):
        raise ValueError("probability/policy columns in regression batch")
    raw = numeric(frame["y_pred_raw"], "raw prediction")
    release = numeric(frame["y_pred_release"], "released prediction")
    if not np.array_equal(raw, release):
        raise ValueError("raw regression variant differs from released prediction")


def validate_outcomes(series):
    present = series.notna()
    # An all-missing column can have object dtype; it is still an unknown outcome.
    if present.any():
        numeric(series.loc[present], "regression outcomes")
    return series.astype("Float64")


def validate_options(batch, config, *, reference=None, labels=None):
    regression = batch.task == "regression"
    if not regression:
        if any(
            getattr(config, name, None) is not None
            for name in ("rmse_warning", "mae_warning", "abs_bias_warning")
        ):
            raise ValueError("regression thresholds cannot monitor classification")
        return
    unit = getattr(config, "target_unit", None)
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("regression monitoring requires an explicit target_unit")
    if (
        getattr(config, "brier_warning", None) is not None
        or getattr(config, "recall_floor", None) is not None
    ):
        raise ValueError("probability/recall thresholds are invalid for regression")
    if getattr(batch, "summary", {}).get("target_unit") not in (None, unit):
        raise ValueError("request units differ from frozen units")
    if reference is not None and reference.get("target_unit") != unit:
        raise ValueError("reference target unit differs from request")
    if labels is not None and labels["metadata"].get("target_unit") != unit:
        raise ValueError("label target unit differs from request")


def evaluate_predictions(
    predictions, labels, key, *, as_of, min_labelled, min_positives=None, min_negatives=None
):
    required = {key, "y_true", "eligible_at", "observed_at"}
    if (
        not required <= set(labels)
        or key not in predictions
        or labels[key].isna().any()
        or predictions[key].isna().any()
        or labels[key].duplicated().any()
        or predictions[key].duplicated().any()
        or set(labels[key]) != set(predictions[key])
    ):
        raise ValueError("exact unique keyed prediction/label coverage required")
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("as_of requires an explicit timezone")
    joined = predictions.merge(
        labels[[key, "y_true", "eligible_at", "observed_at"]], on=key, validate="one_to_one"
    )
    joined["y_true"] = validate_outcomes(joined.y_true)

    def timestamp(value):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("outcome timestamps require explicit timezones")
        return stamp.tz_convert("UTC")

    if not joined.y_true.notna().equals(joined.observed_at.notna()):
        raise ValueError("observed_at required exactly when outcome is present")
    eligible = joined.eligible_at.map(timestamp)
    observed = pd.to_datetime(
        joined.observed_at.map(lambda v: timestamp(v) if pd.notna(v) else pd.NaT), utc=True
    )
    mature = eligible <= cutoff
    known = joined.y_true.notna() & (observed <= cutoff)
    chosen = joined.loc[mature & known]
    n, n_mature = len(chosen), int(mature.sum())
    result = {
        "task": "regression",
        "n_scored": len(joined),
        "n_mature": n_mature,
        "n_known_as_of": int(known.sum()),
        "n_evaluated": n,
        "mature_label_coverage": n / n_mature if n_mature else None,
        "sufficient_support": n >= min_labelled,
        "score_metrics": {},
        "target_summary": None,
        "population": "mature rows with outcomes observed by as_of",
        "bias_definition": "mean(y_pred - y_true)",
        "residual_definition": "y_true - y_pred",
    }
    # Validate ALL saved predictions even when no outcomes have matured.
    for column in SCORE_COLUMNS:
        numeric(joined[column], column)
    if not n:
        return result
    y = numeric(chosen.y_true, "regression outcomes")
    result["target_summary"] = {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "minimum": float(y.min()),
        "maximum": float(y.max()),
    }
    for column in SCORE_COLUMNS:
        predicted = numeric(chosen[column], column)
        residual = y - predicted
        with np.errstate(over="ignore", invalid="ignore"):
            mse = float(np.mean(residual**2))
            denominator = float(np.sum((y - y.mean()) ** 2))
            r2 = (
                float(1 - np.sum(residual**2) / denominator) if n >= 2 and denominator > 0 else None
            )
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(-residual.mean()),
            "r2": r2,
            "residual_quantiles": {
                str(q): float(np.quantile(residual, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
        }
        flat = [metrics[name] for name in ("rmse", "mae", "bias")]
        if not np.isfinite(flat).all() or (r2 is not None and not np.isfinite(r2)):
            raise ValueError("regression metrics overflowed")
        result["score_metrics"][column] = metrics
    return result


def findings(config, *, drift, quality, performance, reference_n, sampling):
    result = []

    def add(rule, severity, message, evidence):
        result.append(
            {
                "rule": rule,
                "severity": severity,
                "message": message,
                "evidence": evidence,
                "action": "investigate; no automatic refit or promotion",
            }
        )

    enough = min(quality["n_rows"], reference_n) >= config.min_rows
    if not enough:
        add(
            "sample_support",
            "info",
            "Distribution warnings suppressed: insufficient rows",
            {"current": quality["n_rows"], "reference": reference_n},
        )
    for item in quality["features"]:
        if item.get("outside_range"):
            add("quality_range", "warning", "Values outside configured range", item)
        rate = item["missing"] / item["n_rows"] if item["n_rows"] else 0
        if rate > config.missing_rate_warning:
            add(
                "missingness",
                "warning",
                "Source missingness exceeds configured limit",
                {**item, "missing_rate": rate, "limit": config.missing_rate_warning},
            )
    if enough:
        for item in drift:
            if item["js_divergence"] >= config.js_warning:
                add("distribution_change", "warning", "Frozen-bin distribution changed", item)
            if item.get("missing_rate_delta", 0) > config.missing_rate_delta_warning:
                add("missingness_change", "warning", "Source missingness increased", item)
    if performance is None:
        add("no_labels", "info", "No outcomes selected; predictive performance is unknown", {})
        return result
    coverage = performance["mature_label_coverage"]
    if coverage is None or coverage < config.min_label_coverage:
        add(
            "label_coverage",
            "warning",
            "Mature outcomes incompletely labelled",
            {"coverage": coverage},
        )
    if sampling != "all_rows":
        add(
            "label_selection",
            "warning",
            "Outcome collection is selected or unknown",
            {"sampling": sampling},
        )
    if not performance["sufficient_support"]:
        add(
            "outcome_support",
            "info",
            "Performance warnings suppressed: insufficient outcomes",
            {"n_evaluated": performance["n_evaluated"], "minimum": config.min_labelled},
        )
    eligible = (
        performance["sufficient_support"]
        and coverage is not None
        and coverage >= config.min_label_coverage
        and sampling == "all_rows"
    )
    if eligible:
        metrics = performance["score_metrics"].get("y_pred_release", {})
        for metric, field in (
            ("rmse", "rmse_warning"),
            ("mae", "mae_warning"),
            ("bias", "abs_bias_warning"),
        ):
            limit, value = getattr(config, field), metrics.get(metric)
            if limit is not None and value is not None and abs(value) > limit:
                add(
                    field,
                    "warning",
                    "Regression performance exceeds configured limit",
                    {
                        "metric": metric,
                        "value": value,
                        "limit": limit,
                        "target_unit": config.target_unit,
                    },
                )
    return result
