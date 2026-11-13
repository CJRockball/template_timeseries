"""Exploratory OOF threshold diagnostics. No fitting, I/O or policy freezing.

Predictions use p >= threshold. Candidates include every unique score, 0, 1,
and nextafter(1, inf) for the flag-none policy when a score is exactly one.
Tied objective maxima choose the highest candidate threshold. Near-best ranges
are envelopes over candidates, not confidence intervals or guaranteed plateaus.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NONE_THRESHOLD = float(np.nextafter(1.0, np.inf))


def _prepare(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and scores must be nonempty aligned 1-D arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be non-null binary 0/1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return y.astype(np.int64), p


def _counts(tp, fp, fn, tn):
    tp, fp, fn, tn = [np.asarray(a, dtype=np.int64) for a in (tp, fp, fn, tn)]
    n = tp + fp + fn + tn

    def ratio(num, den):
        return np.divide(num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=den != 0)

    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "flagged": tp + fp,
        "alert_rate": ratio(tp + fp, n),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
    }
    for beta in (1, 2):
        b = beta * beta
        score = ratio((1 + b) * tp, (1 + b) * tp + b * fn + fp)
        # With positives but no detections F is zero; no positive labels => undefined.
        result[f"f{beta}"] = np.where(tp + fn > 0, np.nan_to_num(score, nan=0.0), np.nan)
    return result


def operating_point(y_true, probabilities, threshold: float) -> dict:
    y, p = _prepare(y_true, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= NONE_THRESHOLD:
        raise ValueError("Threshold must be in [0, 1], or nextafter(1, inf) for flag-none")
    flags, positive = p >= threshold, y == 1
    counts = _counts(
        np.sum(flags & positive),
        np.sum(flags & ~positive),
        np.sum(~flags & positive),
        np.sum(~flags & ~positive),
    )
    return {"threshold": float(threshold), **{k: v.item() for k, v in counts.items()}}


def threshold_profile(y_true, probabilities) -> pd.DataFrame:
    """Exact distinct-score sweep; O(n log n), not an arbitrary 0.01 search grid."""
    y, p = _prepare(y_true, probabilities)
    order = np.argsort(p, kind="stable")
    scores = p[order]
    prefix = np.r_[0, np.cumsum(y[order])]
    candidates = np.unique(np.r_[0.0, scores, 1.0, NONE_THRESHOLD])
    left = np.searchsorted(scores, candidates, side="left")
    flagged = len(y) - left
    tp = prefix[-1] - prefix[left]
    fp = flagged - tp
    fn = prefix[-1] - tp
    tn = len(y) - prefix[-1] - fp
    return pd.DataFrame({"threshold": candidates, **_counts(tp, fp, fn, tn)})


def recommendations(profile: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Max F1/F2 and absolute-score near-best envelopes. No positive labels => no optimum."""
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Near-best absolute tolerance must be in [0, 1]")
    rows = []
    for objective in ("f1", "f2"):
        valid = profile.dropna(subset=[objective]).sort_values("threshold")
        if valid.empty:
            rows.append(
                {
                    "objective": objective,
                    "status": "undefined: no positive labels",
                    "threshold": np.nan,
                    "objective_value": np.nan,
                    "n_positive": 0,
                    "n_negative": int(profile.iloc[0]["fp"] + profile.iloc[0]["tn"]),
                }
            )
            continue
        best = float(valid[objective].max())
        maxima = valid[np.isclose(valid[objective], best, atol=1e-12, rtol=1e-12)]
        choice = maxima.iloc[-1]
        near = valid[objective] >= best - tolerance - 1e-12
        regions = int((near & ~near.shift(fill_value=False)).sum())
        support = valid.loc[near, "threshold"]
        positive, negative = int(choice["tp"] + choice["fn"]), int(choice["fp"] + choice["tn"])
        rows.append(
            {
                **choice.to_dict(),
                "objective": objective,
                "objective_value": best,
                "status": "exploratory OOF" if negative else "exploratory: no negative labels",
                "n_positive": positive,
                "n_negative": negative,
                "near_candidate_min": float(support.min()),
                "near_candidate_max": float(support.max()),
                "near_regions": regions,
                "absolute_tolerance": tolerance,
                "flag_none": bool(choice["threshold"] > 1),
            }
        )
    return pd.DataFrame(rows)


def fold_recommendations(y_true, probabilities, folds, tolerance: float = 0.01) -> pd.DataFrame:
    """Separately selected fold optima: descriptive sensitivity, NOT validation of a tuned policy."""
    y, p = _prepare(y_true, probabilities)
    f = np.asarray(folds, dtype=float)
    if (
        f.ndim != 1
        or len(f) != len(y)
        or not np.isfinite(f).all()
        or ((f < 0) | (f != np.floor(f))).any()
    ):
        raise ValueError("Aligned nonnegative integer development folds required")
    parts = []
    for fold in np.unique(f):
        mask = f == fold
        part = recommendations(threshold_profile(y[mask], p[mask]), tolerance)
        part.insert(0, "fold", int(fold))
        part["n_rows"] = int(mask.sum())
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def budget_boundaries(y_true, probabilities, rates=(0.05, 0.10)) -> pd.DataFrame:
    """Rows must be in canonical key order for deterministic tie-breaking.

    A top-k policy can split a boundary tie; one p >= threshold cutoff cannot.
    The boundary_score is a deployable cutoff only when threshold_exact is true.
    """
    y, p = _prepare(y_true, probabilities)
    order = np.argsort(-p, kind="stable")
    rows = []
    for rate in sorted(set(rates)):
        if not np.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("Alert rate must be in (0, 1]")
        k = max(1, math.ceil(round(rate * len(y), 9)))
        boundary = float(p[order[k - 1]])
        flags = np.zeros(len(y), dtype=bool)
        flags[order[:k]] = True
        positive = y == 1
        counts = _counts(
            np.sum(flags & positive),
            np.sum(flags & ~positive),
            np.sum(~flags & positive),
            np.sum(~flags & ~positive),
        )
        scalar = {key: value.item() for key, value in counts.items()}
        threshold_count = int((p >= boundary).sum())
        rows.append(
            {
                "requested_rate": float(rate),
                "boundary_score": boundary,
                "threshold_exact": threshold_count == k,
                "boundary_tie_count": int((p == boundary).sum()),
                "flagged_if_score_ge_boundary": threshold_count,
                **scalar,
            }
        )
    return pd.DataFrame(rows)
