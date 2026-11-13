"""AI4I 2020 constants and formulas shared by the AI4I blocks (SI units).

Thresholds follow the UCI dataset description. Changing anything in this file changes the
code hash of every block in this package, so existing block files are not silently reused.
"""

from __future__ import annotations

import math

import pandas as pd

AIR = "air_temperature_k"
PROCESS = "process_temperature_k"
SPEED = "rotational_speed_rpm"
TORQUE = "torque_nm"
WEAR = "tool_wear_min"
TYPE = "type"
FEATURES = (TYPE, AIR, PROCESS, SPEED, TORQUE, WEAR)

HDF_TEMP_DIFF_K = 8.6  # heat dissipation: temperature difference below this ...
HDF_SPEED_RPM = 1380.0  # ... and speed below this
PWF_LOW_W = 3500.0  # power failure: power below this or above PWF_HIGH_W
PWF_HIGH_W = 9000.0
OSF_LIMIT_MINNM = {"L": 11000.0, "M": 12000.0, "H": 13000.0}  # wear x torque, by type
TWF_WEAR_START_MIN = 200.0  # tool wear window starts here

RPM_TO_RAD_PER_S = 2 * math.pi / 60


def temp_diff_k(df: pd.DataFrame) -> pd.Series:
    return df[PROCESS] - df[AIR]


def power_w(df: pd.DataFrame) -> pd.Series:
    return df[TORQUE] * df[SPEED] * RPM_TO_RAD_PER_S
