"""Frozen-bin descriptive distribution comparisons."""

import json

import numpy as np


def category_token(value):
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, allow_nan=False)


def probabilities(counts):
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("finite, nonnegative, nonempty counts required")
    return values / values.sum()


def js_divergence(a, b):
    p, q = probabilities(a), probabilities(b)
    if p.shape != q.shape:
        raise ValueError("distribution shapes differ")
    midpoint = (p + q) / 2

    def term(values):
        mask = values > 0
        return np.sum(values[mask] * np.log2(values[mask] / midpoint[mask]))

    return float((term(p) + term(q)) / 2)


def profile(series, *, numeric, bins=10):
    if series.empty or series.isna().any():
        raise ValueError("nonempty non-null reference required")
    if numeric:
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("finite numeric reference required")
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))).tolist()
        counts = np.bincount(np.searchsorted(cuts, values, side="right"), minlength=len(cuts) + 1)
        return {
            "kind": "numeric",
            "cuts": cuts,
            "counts": counts.tolist(),
            "n_rows": len(values),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
        }
    tokens = series.map(category_token)
    categories = sorted(tokens.unique().tolist())
    counts = [int(tokens.eq(value).sum()) for value in categories] + [0]
    return {
        "kind": "categorical",
        "categories": categories,
        "counts": counts,
        "n_rows": len(series),
    }


def compare(series, reference):
    if series.empty or series.isna().any():
        raise ValueError("nonempty non-null current distribution required")
    if reference["kind"] == "numeric":
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite current values")
        counts = np.bincount(
            np.searchsorted(reference["cuts"], values, side="right"),
            minlength=len(reference["counts"]),
        )
        extra = {
            "outside_reference_range": int(
                ((values < reference["minimum"]) | (values > reference["maximum"])).sum()
            )
        }
    else:
        tokens = series.map(category_token)
        categories = reference["categories"]
        counts = [int(tokens.eq(value).sum()) for value in categories] + [
            int((~tokens.isin(categories)).sum())
        ]
        extra = {"unseen_reference_categories": int(counts[-1])}
    return {
        "js_divergence": js_divergence(reference["counts"], counts),
        "n_current": len(series),
        "n_reference": reference["n_rows"],
        "current_counts": np.asarray(counts).tolist(),
        "reference_counts": reference["counts"],
        "method": "frozen_bin_js_bits",
        **extra,
    }
