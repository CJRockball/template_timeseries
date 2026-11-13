"""Finite split-conformal state and independent affine point corrections.

Coverage is marginal under exchangeability, not a conditional/subgroup guarantee.
Ordinary OOF residuals are not a split-conformal calibration population.
"""

import numpy as np


def vector(values, name):
    v = np.asarray(values)
    if v.ndim != 1 or v.dtype.kind not in "iuf" or not np.isfinite(v).all():
        raise ValueError(name + " must be a finite real numeric vector")
    return v.astype(float)


def coverage_tag(level):
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not np.isfinite(level)
        or not 0 < level < 1
    ):
        raise ValueError("coverage must be strictly between zero and one")
    k = round(level * 1000000)
    if not 0 < k < 1000000 or abs(k / 1000000 - level) > 1e-12:
        raise ValueError("coverage supports at most six decimal places")
    return "c" + str(k)


def levels(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("explicit ordered coverage list required")
    tags = [coverage_tag(v) for v in values]
    if len(set(tags)) != len(tags) or list(values) != sorted(values):
        raise ValueError("coverages must be unique and increasing")
    return [float(v) for v in values]


def rank_for(n, coverage):
    if type(n) is not int or n < 0:
        raise ValueError("nonnegative integer calibration size required")
    return ((n + 1) * int(coverage_tag(coverage)[1:]) + 999999) // 1000000


def validate_correction(c):
    if not isinstance(c, dict) or set(c) != {"method", "slope", "intercept"}:
        raise ValueError("invalid correction fields")
    if c["method"] not in {"none", "bias", "affine"}:
        raise ValueError("unsupported correction method")
    for name in ("slope", "intercept"):
        if (
            isinstance(c[name], bool)
            or not isinstance(c[name], (int, float))
            or not np.isfinite(c[name])
        ):
            raise ValueError("invalid correction value")
    if c["method"] in {"none", "bias"} and c["slope"] != 1:
        raise ValueError("identity/bias correction requires unit slope")
    if c["method"] == "none" and c["intercept"] != 0:
        raise ValueError("identity correction requires zero intercept")
    return c


def fit_correction(predicted, target, method):
    p, y = vector(predicted, "predictions"), vector(target, "targets")
    if not len(p) or len(p) != len(y):
        raise ValueError("nonempty aligned recalibration vectors required")
    if method == "none":
        slope, intercept = 1.0, 0.0
    elif method == "bias":
        slope, intercept = 1.0, float(np.mean(y - p))
    elif method == "affine":
        centered = p - p.mean()
        denominator = float(np.dot(centered, centered))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("affine correction requires varying predictions")
        slope = float(np.dot(centered, y - y.mean()) / denominator)
        intercept = float(y.mean() - slope * p.mean())
    else:
        raise ValueError("unsupported correction")
    return validate_correction({"method": method, "slope": slope, "intercept": intercept})


def correct(predicted, correction):
    c = validate_correction(correction)
    with np.errstate(over="ignore", invalid="ignore"):
        result = vector(predicted, "predictions") * c["slope"] + c["intercept"]
    return vector(result, "corrected predictions")


def fit_quantiles(predicted, target, coverages):
    p, y = vector(predicted, "calibration predictions"), vector(target, "calibration targets")
    requested = levels(coverages)
    if len(p) != len(y) or (requested and not len(y)):
        raise ValueError("nonempty aligned conformal calibration vectors required")
    scores = np.sort(np.abs(y - p))
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite residual scores")
    result = {}
    for coverage in requested:
        rank = rank_for(len(scores), coverage)
        if rank > len(scores):
            raise ValueError("sample too small for a finite requested interval")
        result[coverage_tag(coverage)] = {
            "coverage": coverage,
            "rank": rank,
            "half_width": float(scores[rank - 1]),
        }
    return result


def validate_parameters(state):
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "correction",
        "n_calibration",
        "quantiles",
    }:
        raise ValueError("unexpected parameter fields")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise ValueError("unsupported parameter schema")
    validate_correction(state["correction"])
    n = state["n_calibration"]
    if type(n) is not int or n < 0 or not isinstance(state["quantiles"], dict):
        raise ValueError("invalid sample metadata")
    for item in state["quantiles"].values():
        if not isinstance(item, dict) or set(item) != {"coverage", "rank", "half_width"}:
            raise ValueError("invalid quantile fields")
        if state["quantiles"].get(coverage_tag(item["coverage"])) != item:
            raise ValueError("coverage tag mismatch")
        if (
            type(item["rank"]) is not int
            or item["rank"] != rank_for(n, item["coverage"])
            or not 1 <= item["rank"] <= n
        ):
            raise ValueError("invalid finite sample rank")
        q = item["half_width"]
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not np.isfinite(q) or q < 0:
            raise ValueError("invalid half-width")
    widths = [
        i["half_width"] for i in sorted(state["quantiles"].values(), key=lambda i: i["coverage"])
    ]
    if widths != sorted(widths):
        raise ValueError("intervals must be nested")
    return state


def interval_columns(state, selected):
    validate_parameters(state)
    result = {}
    for coverage in levels(selected):
        tag = coverage_tag(coverage)
        if tag not in state["quantiles"]:
            raise ValueError("coverage was not calibrated")
        result[tag] = {"coverage": coverage, "lower": "pi_lower_" + tag, "upper": "pi_upper_" + tag}
    return result


def apply(predicted, state, selected):
    validate_parameters(state)
    point = correct(predicted, state["correction"])
    bounds = {}
    for tag, item in interval_columns(state, selected).items():
        q = state["quantiles"][tag]["half_width"]
        bounds[item["lower"]] = vector(point - q, "lower bound")
        bounds[item["upper"]] = vector(point + q, "upper bound")
    return point, bounds


def interval_metrics(target, columns, definitions):
    y = vector(target, "outcomes")
    result = {}
    for tag, item in definitions.items():
        lower, upper = (
            vector(columns[item["lower"]], "lower bound"),
            vector(columns[item["upper"]], "upper bound"),
        )
        if lower.shape != y.shape or upper.shape != y.shape or (lower > upper).any():
            raise ValueError("misaligned or inverted intervals")
        width = vector(upper - lower, "interval width")
        result[tag] = {
            "nominal_coverage": item["coverage"],
            "n_evaluated": len(y),
            "empirical_coverage": float(np.mean((lower <= y) & (y <= upper))) if len(y) else None,
            "mean_width": float(width.mean()) if len(y) else None,
        }
    return result
