"""Asset registries and feature-set metadata; never load row-level target data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ml_template.dashboard.inventory_common import (
    file_state,
    json_list,
    read_registry,
    resolve_file,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine


def load_assets(engine: Engine, repo_root: Path) -> dict[str, pd.DataFrame]:
    tables = {
        name: read_registry(engine, name)
        for name in ("silver_datasets", "feature_sets", "feature_blocks")
    }
    datasets = tables["silver_datasets"].copy()
    datasets["parquet_file"] = datasets["parquet_path"].map(lambda p: file_state(p, repo_root))
    datasets["manifest_file"] = datasets["manifest_path"].map(lambda p: file_state(p, repo_root))
    tables["silver_datasets"] = datasets
    return tables


def dataset_details(row: dict, repo_root: Path) -> dict:
    """Manifest metadata only. Does not recompute quality, prevalence or file hashes."""
    path = resolve_file(row.get("manifest_path"), repo_root)
    if path is None:
        return {"message": "Registered manifest is missing or unavailable"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: manifest.get(key)
        for key in (
            "silver_dataset_version_id",
            "row_count",
            "columns",
            "roles",
            "validation_passed",
            "validation_report",
            "parquet_sha256",
        )
    }


def feature_columns(sets: pd.DataFrame, blocks: pd.DataFrame, set_id: str | None) -> pd.DataFrame:
    columns = ["position", "feature", "source", "source_status"]
    found = sets[sets["feature_set_id"] == set_id]
    if found.empty:
        return pd.DataFrame(columns=columns)
    row = found.iloc[0]
    sources = {}
    missing = False
    for block_id in json_list(row["blocks_json"]):
        block = blocks[blocks["block_id"] == block_id]
        if block.empty:
            missing = True
            continue
        for name in json_list(block.iloc[0]["columns_json"]):
            sources[name] = str(block_id)
    return pd.DataFrame(
        [
            {
                "position": i + 1,
                "feature": name,
                "source": sources.get(name, "unresolved" if missing else "Silver"),
                "source_status": "verify missing block registry"
                if missing and name not in sources
                else "registered",
            }
            for i, name in enumerate(json_list(row["columns_json"]))
        ],
        columns=columns,
    )


def lineage_table(sets: pd.DataFrame) -> pd.DataFrame:
    out = sets[
        [
            "feature_set_id",
            "parent_feature_set_id",
            "silver_dataset_version_id",
            "n_features",
            "description",
        ]
    ].copy()
    ids = set(sets["feature_set_id"])
    out["parent_status"] = out["parent_feature_set_id"].map(
        lambda parent: (
            "root"
            if parent is None or pd.isna(parent)
            else ("registered" if parent in ids else "missing registry row")
        )
    )
    return out


def ranked_models(runs: pd.DataFrame, higher: bool = True) -> pd.DataFrame:
    """Rank successful CV model fits within a single explicitly selected cohort."""
    cohort = ["silver_dataset_version_id", "split_id", "primary_metric"]
    if runs[cohort].drop_duplicates().shape[0] > 1:
        raise ValueError("Select one dataset, split and metric before ranking models")
    columns = [
        "rank",
        "run_id",
        "experiment_name",
        "experiment_run_id",
        "seed",
        "model_type",
        "feature_set_id",
        "primary_metric",
        "primary_value",
        "n_features",
        "n_rounds_final",
        "duration_s",
        "started_at",
    ]
    selected = runs[(runs["status"] == "succeeded") & runs["primary_value"].notna()].copy()
    selected = selected.sort_values(["primary_value", "run_id"], ascending=[not higher, True])
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected.reindex(columns=columns)
