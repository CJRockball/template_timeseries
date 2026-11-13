"""Pure summaries of a declared explanation population; no fitting or selection."""

import numpy as np
import pandas as pd

from ml_template.explainability.result import MarginExplanation


def feature_summary(explanation: MarginExplanation) -> pd.DataFrame:
    values = explanation.contributions
    out = pd.DataFrame(
        {
            "feature": values.columns,
            "mean_abs_contribution": values.abs().mean().to_numpy(),
            "mean_contribution": values.mean().to_numpy(),
            "positive_fraction": (values > 0).mean().to_numpy(),
            "n_rows": len(values),
            "output_scale": explanation.output_scale,
        }
    )
    return out.sort_values(
        ["mean_abs_contribution", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def grouped_summary(explanation: MarginExplanation, groups: dict) -> pd.DataFrame:
    """Require an exact non-overlapping partition; aggregate signed values before abs.

    This is a post-hoc grouping of feature contributions, not recomputed group SHAP.
    """
    if not isinstance(groups, dict) or not groups:
        raise ValueError("nonempty group mapping required")
    assigned = []
    for name, columns in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("group names must be nonempty strings")
        if not isinstance(columns, (list, tuple)) or not columns:
            raise ValueError("each group must contain a nonempty feature list")
        if not all(isinstance(c, str) for c in columns):
            raise ValueError("group feature names must be strings")
        assigned.extend(columns)
    if len(set(assigned)) != len(assigned):
        raise ValueError("feature groups overlap")
    if set(assigned) != set(explanation.contributions.columns):
        raise ValueError("groups must partition every explained feature exactly once")
    records = []
    for name, columns in groups.items():
        signed = explanation.contributions[list(columns)].sum(axis=1)
        records.append(
            {
                "group": name,
                "mean_abs_contribution": float(np.abs(signed).mean()),
                "mean_contribution": float(signed.mean()),
                "n_features": len(columns),
                "n_rows": len(signed),
                "output_scale": explanation.output_scale,
                "aggregation": "sum_signed_features_then_mean_absolute",
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["mean_abs_contribution", "group"], ascending=[False, True])
        .reset_index(drop=True)
    )
