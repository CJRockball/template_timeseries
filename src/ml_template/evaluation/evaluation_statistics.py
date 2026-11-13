"""Pure report metrics; pr_auc means average precision."""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ml_template.evaluation.decision_policy import (
    apply_policy,
    labels,
    policy_metrics,
    probabilities,
)
from ml_template.evaluation.evaluation_config import PolicyConfig


def score_metrics(y, p, rates=()):
    p = probabilities(p)
    y = labels(y, len(p))
    both = len(np.unique(y)) == 2
    out = {
        "n_rows": len(y),
        "n_positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
    }
    for r in rates:
        panel = policy_metrics(
            y,
            apply_policy(
                p, PolicyConfig(kind="alert_rate", value=r, source="predeclared diagnostic")
            ),
        )
        for m in ["recall", "precision", "lift"]:
            out[f"{m}_at_{round(r * 100, 6):g}pct"] = panel[m]
    return out


def variant_metrics(y, variants, cfg):
    out = {}
    for v, p in variants.items():
        out[v] = score_metrics(y, p, cfg.alert_rates)
        if v == cfg.release_variant and cfg.policy.kind != "none":
            out[v].update(
                {
                    f"policy_{k}": val
                    for k, val in policy_metrics(y, apply_policy(p, cfg.policy)).items()
                }
            )
    return out


def assess_criteria(rows, cfg):
    metrics = {r["metric"]: r for r in rows if r["variant"] == cfg.release_variant}
    details = []
    for c in cfg.criteria:
        key = "value" if c.basis == "point" else ("ci_low" if c.operator == ">=" else "ci_high")
        value = metrics.get(c.metric, {}).get(key)
        met = value is not None and (value >= c.value if c.operator == ">=" else value <= c.value)
        details.append(
            {
                **c.model_dump(),
                "observed": value,
                "status": ("met" if met else "not_met") if value is not None else "not_assessed",
            }
        )
    states = {d["status"] for d in details}
    status = "not_met" if "not_met" in states else ("met" if states == {"met"} else "not_assessed")
    return {"status": status, "criteria": details, "automatic_promotion": False}
