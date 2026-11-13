"""Block ``thermal_power``: temperature difference and mechanical power.

Direct inputs to the heat dissipation and power failure rules.
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
    SPEED,
    TORQUE,
    power_w,
    temp_diff_k,
)


def _sanity(df: pd.DataFrame, out: pd.DataFrame, params: Mapping[str, Any]) -> dict[str, float]:
    rule = (out["temp_diff_k"] < HDF_TEMP_DIFF_K) & (df[SPEED] < HDF_SPEED_RPM)
    return {"hdf_rule_rows": float(rule.sum())}


@register_block(
    "thermal_power",
    requires=(AIR, PROCESS, SPEED, TORQUE),
    description="temp_diff_k = process - air (K); power_w = torque x speed in rad/s (W)",
    sanity=_sanity,
    expected={"hdf_rule_rows": 115.0},  # full AI4I 2020 table, 10,000 rows
)
def thermal_power(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"temp_diff_k": temp_diff_k(df), "power_w": power_w(df)}, index=df.index)
