"""Reserve calibration roles only from original non-holdout rows."""

import math

import numpy as np


def reserve(n_rows, development_indices, config):
    indices = np.sort(np.asarray(development_indices, dtype=int))
    partition = np.full(n_rows, "holdout", dtype=object)
    partition[indices] = "dev"
    fractions = [
        getattr(config, "recalibration_fraction", 0),
        getattr(config, "calibration_fraction", 0),
    ]
    if any(fractions):
        if config.method != "random":
            raise ValueError("regression reservations require random splits")
        counts = [math.ceil(n_rows * fraction) for fraction in fractions]
        if sum(counts) > len(indices) - config.n_folds:
            raise ValueError("insufficient development rows after calibration reservation")
        shuffled = np.random.RandomState(config.seed).permutation(indices)
        offset = 0
        for role, count in zip(("recalibration", "calibration"), counts, strict=True):
            partition[shuffled[offset : offset + count]] = role
            offset += count
    return np.flatnonzero(partition == "dev"), partition


def validate_assignments(config, assignments, manifest):
    allowed = {"dev", "holdout"}
    for role in ("calibration", "recalibration"):
        expected = math.ceil(len(assignments) * getattr(config, role + "_fraction", 0))
        if expected:
            allowed.add(role)
        if int(assignments.partition.eq(role).sum()) != expected or (
            expected and manifest.get("n_" + role) != expected
        ):
            raise ValueError("reserved count differs from policy: " + role)
    if not assignments.partition.isin(allowed).all():
        raise ValueError("unexpected partition")
