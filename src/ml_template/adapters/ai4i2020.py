# src/ml_template/adapters/ai4i.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

AI4I_REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}

COLUMN_RENAMES = {
    "UDI": "entity_id",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "target",
}


def load_and_normalize(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = AI4I_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"AI4I source is missing required columns: {sorted(missing)}")

    frame = frame.rename(columns=COLUMN_RENAMES).copy()
    frame["entity_id"] = frame["entity_id"].astype("int64")
    frame["target"] = frame["target"].astype("int8")

    return frame
