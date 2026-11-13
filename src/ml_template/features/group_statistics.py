"""Fitted group statistics: per-group mean, std and similar, learned on training rows only.

For each group (a combination of categorical columns) the step learns statistics of numeric
columns, then adds them to every row, optionally with the row's deviation from its group mean.
No target is used. Groups never seen in training, or smaller than ``min_group_size``, fall back
to the overall training statistics. A row contributes to its own group's statistic, so keep
``min_group_size`` above 1 for small groups.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

STATS = ("mean", "std", "median", "min", "max")


class GroupStatisticsStep:
    name = "group_statistics"

    def __init__(
        self,
        group_by: Sequence[str],
        columns: Sequence[str],
        stats: Sequence[str] = ("mean", "std"),
        deviation: bool = False,
        zscore: bool = False,
        count: bool = False,
        frequency: bool = False,
        min_group_size: int = 1,
    ) -> None:
        if not group_by or not columns:
            raise ValueError("group_statistics needs group_by and columns")
        unknown = sorted(set(stats) - set(STATS))
        if unknown:
            raise ValueError(f"unknown stats {unknown}; available: {list(STATS)}")
        if min_group_size < 1:
            raise ValueError("min_group_size must be at least 1")
        self.group_by = list(group_by)
        self.columns = list(columns)
        self.stats = list(stats)
        self.deviation = deviation
        self.zscore = zscore
        self.count = count
        self.frequency = frequency
        self.min_group_size = int(min_group_size)
        self._tag = "_".join(self.group_by)

    def _label(self, col: str, suffix: str) -> str:
        return f"{col}__by_{self._tag}__{suffix}"

    def output_columns(self) -> list[str]:
        cols: list[str] = []
        for c in self.columns:
            cols += [self._label(c, s) for s in self.stats]
            if self.deviation:
                cols.append(self._label(c, "dev"))
            if self.zscore:
                cols.append(self._label(c, "z"))
        if self.count:
            cols.append(f"grp_{self._tag}__count")
        if self.frequency:
            cols.append(f"grp_{self._tag}__freq")
        return cols

    def _key(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in [*self.group_by, *self.columns] if c not in X.columns]
        if missing:
            raise KeyError(f"group_statistics needs columns {missing}")
        key = X[self.group_by[0]].astype(str)
        for col in self.group_by[1:]:
            key = key + "|" + X[col].astype(str)
        return key.to_numpy()

    def fit(self, X: pd.DataFrame) -> GroupStatisticsStep:
        key = self._key(X)
        num = X[self.columns].astype(float)
        grouped = num.groupby(key)
        table = grouped.agg(list(STATS))
        table.columns = [f"{c}__{s}" for c, s in table.columns]
        overall = num.agg(list(STATS))
        self.table_ = table
        self.size_ = grouped.size()
        self.global_ = {f"{c}__{s}": float(overall.loc[s, c]) for c in self.columns for s in STATS}
        self.n_ = len(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        key = self._key(X)
        size = self.size_.reindex(key).fillna(0).to_numpy()
        look = self.table_.reindex(key).reset_index(drop=True)
        small = size < self.min_group_size
        for col in look.columns:
            look[col] = look[col].where(~small, self.global_[col])
        look = look.fillna(self.global_)
        look.index = X.index
        out: dict[str, pd.Series | np.ndarray] = {}
        for c in self.columns:
            for s in self.stats:
                out[self._label(c, s)] = look[f"{c}__{s}"]
            x = X[c].astype(float)
            if self.deviation:
                out[self._label(c, "dev")] = x - look[f"{c}__mean"]
            if self.zscore:
                std = look[f"{c}__std"]
                out[self._label(c, "z")] = ((x - look[f"{c}__mean"]) / std).where(std > 0, 0.0)
        if self.count:
            out[f"grp_{self._tag}__count"] = pd.Series(size, index=X.index)
        if self.frequency:
            out[f"grp_{self._tag}__freq"] = pd.Series(size / self.n_, index=X.index)
        return pd.DataFrame(out, index=X.index)
