"""Support-aware diagnostics; no fitting or policy selection."""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from ml_template.evaluation.decision_policy import policy_metrics
from ml_template.evaluation.evaluation_statistics import score_metrics


def reliability_rows(y, variants, bins):
    from ml_template.evaluation.calibration_diagnostics import reliability_table

    rows = []
    for v, p in variants.items():
        rows.extend(
            {"variant": v, **r} for r in reliability_table(y, p, bins).to_dict(orient="records")
        )
    return rows


def curve_rows(y, variants):
    y = np.asarray(y)
    rows = []
    for v, p in variants.items():
        if len(np.unique(y)) == 2:
            precision, recall, _ = precision_recall_curve(y, p)
            rows.extend(
                {"kind": "precision_recall", "variant": v, "x": float(x), "y": float(z)}
                for x, z in zip(recall, precision, strict=True)
            )
            fpr, tpr, _ = roc_curve(y, p)
            rows.extend(
                {"kind": "roc", "variant": v, "x": float(x), "y": float(z)}
                for x, z in zip(fpr, tpr, strict=True)
            )
        cum = np.r_[0, np.cumsum(y[np.argsort(-np.asarray(p), kind="stable")])]
        idx = np.unique(np.r_[0, np.linspace(1, len(y), min(len(y), 500), dtype=int)])
        rows.extend(
            {
                "kind": "capture",
                "variant": v,
                "x": float(i / len(y)),
                "y": float(cum[i] / y.sum()) if y.sum() else None,
            }
            for i in idx
        )
    return rows


def subgroup_rows(X, y, variants, flags, cfg):
    y = np.asarray(y)
    rows = []
    for s in cfg.slices:
        for val in s.values:
            mask = X[s.column].eq(val).fillna(False).to_numpy(dtype=bool)
            n, pos = int(mask.sum()), int(y[mask].sum())
            supported = n >= cfg.min_slice_rows and min(pos, n - pos) >= cfg.min_slice_positives
            for v, p in variants.items():
                row = {
                    "column": s.column,
                    "slice_value": str(val),
                    "variant": v,
                    "n_rows": n,
                    "n_positives": pos,
                    "base_rate": pos / n if n else None,
                    "status": "supported" if supported else "insufficient_support",
                }
                if supported:
                    row.update(score_metrics(y[mask], np.asarray(p)[mask]))
                    if v == cfg.release_variant and flags is not None:
                        row.update(
                            {
                                f"policy_{k}": value
                                for k, value in policy_metrics(y[mask], flags[mask]).items()
                            }
                        )
                rows.append(row)
    return rows


def error_rows(frame, X, flags, cfg):
    if flags is None or not cfg.error_sample_size:
        return []
    rows = []
    for name, mask, asc in [
        ("false_negative", frame.y_true.eq(1).to_numpy() & ~flags, True),
        ("false_positive", frame.y_true.eq(0).to_numpy() & flags, False),
    ]:
        indices = (
            frame.loc[mask, f"p_{cfg.release_variant}"]
            .sort_values(ascending=asc, kind="stable")
            .head(cfg.error_sample_size)
            .index
        )
        for i in indices:
            rows.append(
                {
                    "error_type": name,
                    **frame.loc[i].to_dict(),
                    **{f"feature:{c}": X.iloc[i][c] for c in X.columns},
                }
            )
    return rows
