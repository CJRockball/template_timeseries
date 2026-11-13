"""Missing-aware frozen profiles; delegates observed-value bins to existing drift."""

import numpy as np

from ml_template.monitoring.drift import compare, js_divergence, profile


def nullable_profile(series, *, numeric, bins=10):
    if series.empty:
        raise ValueError("nonempty reference required")
    observed = series.dropna()
    inner = profile(observed, numeric=numeric, bins=bins) if len(observed) else None
    counts = inner["counts"] if inner else [0]
    return {
        "kind": "nullable_v1",
        "numeric": bool(numeric),
        "observed": inner,
        "counts": [*counts, int(series.isna().sum())],
        "n_rows": len(series),
        "missing_rate": float(series.isna().mean()),
    }


def nullable_compare(series, reference):
    if reference.get("kind") != "nullable_v1" or series.empty:
        raise ValueError("nullable profile and nonempty current series required")
    observed = series.dropna()
    inner = reference["observed"]
    extra = {}
    if inner is None:
        if len(observed):
            # Validate domains, but never learn new monitoring bins from current data.
            profile(observed, numeric=reference["numeric"])
        counts = [len(observed)]
        extra["observed_distribution_unavailable"] = True
    elif len(observed):
        measured = compare(observed, inner)
        counts = measured.pop("current_counts")
        for key in ("outside_reference_range", "unseen_reference_categories"):
            if key in measured:
                extra[key] = measured[key]
    else:
        counts = [0] * len(inner["counts"])
    counts = [*counts, int(series.isna().sum())]
    original = reference["counts"]
    if (
        len(original) != len(counts)
        or sum(original) != reference["n_rows"]
        or not np.isfinite(original).all()
        or min(original) < 0
    ):
        raise ValueError("invalid nullable reference counts")
    rate = float(series.isna().mean())
    return {
        "js_divergence": js_divergence(original, counts),
        "n_current": len(series),
        "n_reference": reference["n_rows"],
        "current_counts": counts,
        "reference_counts": original,
        "missing_rate": rate,
        "reference_missing_rate": reference["missing_rate"],
        "missing_rate_delta": rate - reference["missing_rate"],
        "method": "frozen_bin_js_bits_with_missing_bin",
        **extra,
    }
