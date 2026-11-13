"""Rows the model gets most wrong, with their feature values.

Out-of-fold predictions only. The ``error_kind`` column says which: ``missed_positive`` rows are true positives with the
lowest scores; ``false_alarm`` rows are true negatives with the highest scores.
Reading a few dozen of them shows where the model is blind, which is where a new
feature may help.
"""

from __future__ import annotations

import pandas as pd


def error_sample(
    oof: pd.DataFrame, X: pd.DataFrame, y: pd.Series, key_col: str, size: int
) -> pd.DataFrame:
    """``oof`` has columns [key_col, fold, p_oof]. X and y are indexed by the key."""
    df = oof[[key_col, "fold", "p_oof"]].copy()
    df["y_true"] = y.loc[df[key_col]].to_numpy()
    missed = df[df["y_true"] == 1].nsmallest(size, "p_oof").assign(error_kind="missed_positive")
    alarms = df[df["y_true"] == 0].nlargest(size, "p_oof").assign(error_kind="false_alarm")
    out = pd.concat([missed, alarms], ignore_index=True)
    features = X.loc[out[key_col]].reset_index(drop=True)
    clash = set(features.columns) & set(out.columns)
    if clash:
        raise ValueError(f"feature names clash with report columns: {sorted(clash)}")
    return pd.concat([out, features], axis=1)
