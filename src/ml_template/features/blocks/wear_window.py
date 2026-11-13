"""Block ``wear_window``: distance to the start of the tool wear failure window."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ml_template.features.block_registry import register_block
from ml_template.features.blocks._ai4i_common import TWF_WEAR_START_MIN, WEAR


@register_block(
    "wear_window",
    requires=(WEAR,),
    description="twf_wear_margin_min = wear - window start; twf_wear_in_window = wear >= start",
    params={"window_start_min": TWF_WEAR_START_MIN},
)
def wear_window(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    start = float(params["window_start_min"])
    return pd.DataFrame(
        {
            "twf_wear_margin_min": df[WEAR] - start,
            "twf_wear_in_window": (df[WEAR] >= start).astype("int8"),
        },
        index=df.index,
    )
