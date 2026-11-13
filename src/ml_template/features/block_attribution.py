"""Group per-feature importance by the block each feature came from.

Permutation drops of correlated features are not additive, so the group sum is a guide to
where the signal sits, not an exact share. Compare it with the PR-AUC change of the run.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def attribute_importance(
    importance: pd.DataFrame,
    column_sources: Mapping[str, str],
    *,
    feature_col: str = "feature",
    value_col: str = "value",
) -> pd.DataFrame:
    """Return one row per source (``silver`` or a block name), largest sum first.

    ``importance`` is e.g. the pooled rows of ``run_feature_importance``;
    ``column_sources`` is the ``column_sources`` entry of the feature set manifest.
    """
    unknown = sorted(set(importance[feature_col]) - set(column_sources))
    if unknown:
        raise ValueError(f"features without a source: {unknown}")
    grouped = (
        importance.assign(source=importance[feature_col].map(column_sources))
        .groupby("source")[value_col]
        .agg(sum_value="sum", max_value="max", n_features="size")
    )
    return grouped.sort_values("sum_value", ascending=False).reset_index()
