"""Block ``rule_margins``: signed distances to the documented failure thresholds.

A negative heat or power margin means the rule condition is met on that side. Margins are
continuous, so a tree model can split on how close a row is to a boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ml_template.features.block_registry import register_block
from ml_template.features.blocks._ai4i_common import (
    AIR,
    HDF_SPEED_RPM,
    HDF_TEMP_DIFF_K,
    PROCESS,
    PWF_HIGH_W,
    PWF_LOW_W,
    SPEED,
    TORQUE,
    power_w,
    temp_diff_k,
)


def _sanity(df: pd.DataFrame, out: pd.DataFrame, params: Mapping[str, Any]) -> dict[str, float]:
    hdf = (out["hdf_temp_margin_k"] < 0) & (out["hdf_speed_margin_rpm"] < 0)
    pwf = (out["pwf_low_margin_w"] < 0) | (out["pwf_high_margin_w"] < 0)
    return {"hdf_rule_rows": float(hdf.sum()), "pwf_rule_rows": float(pwf.sum())}


@register_block(
    "rule_margins",
    requires=(AIR, PROCESS, SPEED, TORQUE),
    description="signed distances to the 8.6 K, 1380 rpm, 3500 W and 9000 W thresholds",
    sanity=_sanity,
    expected={"hdf_rule_rows": 115.0},  # full AI4I 2020 table, 10,000 rows
)
def rule_margins(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    power = power_w(df)
    return pd.DataFrame(
        {
            "hdf_temp_margin_k": temp_diff_k(df) - HDF_TEMP_DIFF_K,
            "hdf_speed_margin_rpm": df[SPEED] - HDF_SPEED_RPM,
            "pwf_low_margin_w": power - PWF_LOW_W,
            "pwf_high_margin_w": PWF_HIGH_W - power,
        },
        index=df.index,
    )
