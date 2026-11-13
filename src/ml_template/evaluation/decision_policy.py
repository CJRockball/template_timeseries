"""Pure policies. Input rows must be in ascending canonical entity-key order."""

import math

import numpy as np


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite, nonempty, 1-D and in [0,1]")
    return p


def labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned non-null binary labels required")
    return y.astype(int)


def apply_policy(p, policy):
    p = probabilities(p)
    if policy.kind == "none":
        return None
    if policy.kind == "threshold":
        return p >= policy.value
    k = max(1, math.ceil(round(policy.value * len(p), 9)))
    flags = np.zeros(len(p), dtype=bool)
    flags[np.argsort(-p, kind="stable")[:k]] = True
    return flags


def policy_metrics(y, flags):
    flags = np.asarray(flags, dtype=bool)
    y = labels(y, len(flags))
    pos = y == 1
    tp, fp = int((pos & flags).sum()), int((~pos & flags).sum())
    fn, tn = int((pos & ~flags).sum()), int((~pos & ~flags).sum())

    def ratio(a, b):
        return a / b if b else None

    precision = ratio(tp, tp + fp)
    base = float(y.mean())
    return {
        "n_flagged": int(flags.sum()),
        "flagged_fraction": float(flags.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fn + fp) if tp + fn else None,
        "f2": ratio(5 * tp, 5 * tp + 4 * fn + fp) if tp + fn else None,
        "false_positive_rate": ratio(fp, fp + tn),
        "lift": precision / base if precision is not None and base else None,
    }


# calibrated-policy update v1
