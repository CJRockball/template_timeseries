"""Block ``strain``: tool wear x torque and its margin to the type-specific limit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ml_template.features.block_registry import register_block
from ml_template.features.blocks._ai4i_common import OSF_LIMIT_MINNM, TORQUE, TYPE, WEAR


def _limits(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    limits = params["osf_limit_minnm"]
    types = df[TYPE].astype(str)
    unknown = sorted(set(types.unique()) - set(limits))
    if unknown:
        raise ValueError(f"no overstrain limit for type values {unknown}")
    return types.map(limits).astype(float)


def _sanity(df: pd.DataFrame, out: pd.DataFrame, params: Mapping[str, Any]) -> dict[str, float]:
    return {"osf_rule_rows": float((out["osf_margin_minnm"] < 0).sum())}


@register_block(
    "strain",
    requires=(TYPE, TORQUE, WEAR),
    description="wear_torque_minnm = wear x torque; osf_margin_minnm = type limit - wear_torque",
    params={"osf_limit_minnm": dict(OSF_LIMIT_MINNM)},
    sanity=_sanity,
)
def strain(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    wear_torque = df[WEAR] * df[TORQUE]
    return pd.DataFrame(
        {
            "wear_torque_minnm": wear_torque,
            "osf_margin_minnm": _limits(df, params) - wear_torque,
        },
        index=df.index,
    )
