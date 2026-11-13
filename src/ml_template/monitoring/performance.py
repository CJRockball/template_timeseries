"""Evaluate saved probabilities and decisions without fitting."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ml_template.evaluation.metrics import compute_metrics
from ml_template.monitoring.artifacts import clean


def probability_metrics(y, scores):
    result = compute_metrics(y, scores, ["pr_auc", "roc_auc", "brier"])
    result["log_loss"] = float(log_loss(y, scores, labels=[0, 1]))
    return clean(result)


def reliability(y, scores, bins=10):
    assignments = np.minimum((scores * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        result.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n_rows": count,
                "mean_probability": float(scores[mask].mean()) if count else None,
                "observed_fraction": float(y[mask].mean()) if count else None,
            }
        )
    return result


def evaluate_predictions(
    predictions, labels, key, *, as_of, min_labelled, min_positives, min_negatives
):
    if (
        predictions[key].duplicated().any()
        or labels[key].duplicated().any()
        or set(predictions[key]) != set(labels[key])
    ):
        raise ValueError("exact keyed prediction/label coverage required")
    joined = predictions.merge(
        labels[[key, "y_true", "eligible_at", "observed_at"]], on=key, validate="one_to_one"
    )
    cutoff = pd.Timestamp(as_of)
    mature = joined.eligible_at <= cutoff
    known = joined.y_true.notna() & (joined.observed_at <= cutoff)
    selected = joined.loc[mature & known]
    n_mature, n = int(mature.sum()), len(selected)
    positives = int(selected.y_true.sum()) if n else 0
    negatives = n - positives
    result = {
        "n_scored": len(joined),
        "n_mature": n_mature,
        "n_known_as_of": int(known.sum()),
        "n_evaluated": n,
        "n_positive": positives,
        "n_negative": negatives,
        "mature_label_coverage": n / n_mature if n_mature else None,
        "prevalence": positives / n if n else None,
        "sufficient_support": n >= min_labelled
        and positives >= min_positives
        and negatives >= min_negatives,
        "score_metrics": {},
        "reliability": {},
        "policy": None,
        "population": "mature rows with labels observed by as_of",
    }
    if not n:
        return result
    y = selected.y_true.to_numpy(dtype=int)
    for name in ("p_raw", "p_calibrated", "p_release"):
        if name in selected:
            scores = selected[name].to_numpy(dtype=float)
            result["score_metrics"][name] = probability_metrics(y, scores)
            result["reliability"][name] = reliability(y, scores)
    if "policy_flag" in selected:
        flags = selected.policy_flag.to_numpy(dtype=bool)
        tp, fp = int(((y == 1) & flags).sum()), int(((y == 0) & flags).sum())
        fn, tn = int(((y == 1) & ~flags).sum()), int(((y == 0) & ~flags).sum())
        result["policy"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "source": "original saved policy_flag; not reranked on labelled subset",
        }
    return clean(result)
