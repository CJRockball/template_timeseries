"""Paired IID-row percentile bootstrap, conditional on the frozen fitted predictor."""

import numpy as np

from ml_template.evaluation.evaluation_statistics import variant_metrics

EXCLUDED = {
    "n_rows",
    "n_positives",
    "policy_tp",
    "policy_fp",
    "policy_tn",
    "policy_fn",
    "policy_n_flagged",
}


def bootstrap_metrics(y, variants, cfg):
    y = np.asarray(y)
    variants = {v: np.asarray(p) for v, p in variants.items()}
    point = variant_metrics(y, variants, cfg)
    samples = {(v, m): [] for v, panel in point.items() for m in panel if m not in EXCLUDED}
    paired = {
        m: []
        for m in point.get("raw", {})
        if m in point.get("calibrated", {}) and m not in EXCLUDED
    }
    repetitions = cfg.uncertainty.repetitions if cfg.uncertainty.enabled else 0
    rng = np.random.default_rng(cfg.uncertainty.seed)
    for _ in range(repetitions):
        idx = np.sort(rng.integers(0, len(y), len(y)))
        trial = variant_metrics(y[idx], {v: p[idx] for v, p in variants.items()}, cfg)
        for (v, m), values in samples.items():
            if trial[v][m] is not None and np.isfinite(trial[v][m]):
                values.append(trial[v][m])
        for m, values in paired.items():
            a, b = trial["raw"][m], trial["calibrated"][m]
            if a is not None and b is not None:
                values.append(b - a)

    def interval(values):
        valid = len(values)
        out = {"ci_low": None, "ci_high": None, "n_valid": valid, "n_invalid": repetitions - valid}
        if repetitions and valid >= max(20, int(0.8 * repetitions)):
            alpha = (1 - cfg.uncertainty.confidence) / 2
            lo, hi = np.quantile(values, [alpha, 1 - alpha])
            out.update(ci_low=float(lo), ci_high=float(hi))
        return out

    rows = []
    for v, panel in point.items():
        for m, value in panel.items():
            ci = (
                interval(samples[v, m])
                if (v, m) in samples
                else {"ci_low": None, "ci_high": None, "n_valid": 0, "n_invalid": 0}
            )
            rows.append({"variant": v, "metric": m, "value": value, **ci})
    differences = []
    for m, values in paired.items():
        a, b = point["raw"][m], point["calibrated"][m]
        differences.append(
            {
                "metric": m,
                "value": b - a if a is not None and b is not None else None,
                **interval(values),
            }
        )
    return rows, differences
