"""Class-weight helper shared by the boosting adapters."""

from __future__ import annotations

import pandas as pd


def resolve_scale_pos_weight(value: float | str, y: pd.Series) -> float:
    """'auto' means negatives over positives in the rows being fitted."""
    if value == "auto":
        positives = int(y.sum())
        if positives == 0:
            raise ValueError("cannot compute scale_pos_weight: no positives in training rows")
        return (len(y) - positives) / positives
    return float(value)
