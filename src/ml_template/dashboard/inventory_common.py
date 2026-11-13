"""Shared read-only helpers for asset and sweep pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import Engine

OPTIONAL_TABLES = {
    "silver_datasets": [
        "silver_dataset_version_id",
        "dataset_version_id",
        "status",
        "created_at",
        "row_count",
        "parquet_path",
        "manifest_path",
        "validation_path",
    ],
    "feature_sets": [
        "feature_set_id",
        "silver_dataset_version_id",
        "parent_feature_set_id",
        "blocks_json",
        "columns_json",
        "n_features",
        "description",
        "manifest_path",
    ],
    "feature_blocks": [
        "block_id",
        "block_name",
        "silver_dataset_version_id",
        "columns_json",
        "derived_from_json",
        "params_json",
        "description",
        "parquet_path",
    ],
    "experiment_launches": [
        "experiment_run_id",
        "experiment_name",
        "kind",
        "n_runs",
        "status",
        "started_at",
        "finished_at",
        "sweep_path",
        "error_summary",
    ],
}


def read_registry(engine: Engine, table: str) -> pd.DataFrame:
    """Never creates or migrates tables. Only allow listed identifiers."""
    from sqlalchemy import inspect, text

    columns = OPTIONAL_TABLES[table]
    if not inspect(engine).has_table(table):
        return pd.DataFrame(columns=columns)
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table}"), conn)


def json_list(value) -> list:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError("Expected a JSON list in the registry")
    return result


def resolve_file(value, repo_root: Path) -> Path | None:
    if value is None or pd.isna(value) or not str(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path if path.is_file() else None


def file_state(value, repo_root: Path) -> str:
    return "present" if resolve_file(value, repo_root) else "missing or unrecorded"


def singapore_times(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ("started_at", "finished_at", "created_at"):
        if col in out:
            converted = pd.to_datetime(out[col], utc=True, errors="coerce")
            out[col + "_sgt"] = converted.dt.tz_convert("Asia/Singapore").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            out = out.drop(columns=col)
    return out


def grid(table: pd.DataFrame, height: int = 320):
    import dash_ag_grid as dag

    rounded = singapore_times(table).round(5)
    # AG Grid must not treat dotted parameter names as nested object lookups.
    clean = rounded.astype(object).where(rounded.notna(), None)
    for col in clean:
        clean[col] = clean[col].map(
            lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
        )

    # Mapping can infer numeric dtype and reintroduce NaN.
    clean = clean.astype(object).where(clean.notna(), None)

    return dag.AgGrid(
        rowData=clean.to_dict("records"),
        columnDefs=[{"field": str(c)} for c in clean],
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": 20,
            "suppressFieldDotNotation": True,
        },
        style={"height": f"{height}px"},
    )
