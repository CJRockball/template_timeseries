"""Replay frozen stateless recipes; never join historical values onto new rows."""

import json
from pathlib import Path

import pandas as pd

from ml_template.features.block_registry import get_block


def replay_recipe(raw_features, recipe):
    if recipe.get("schema_version", 1) != 1:
        raise ValueError("unsupported feature recipe schema")
    if list(raw_features.columns) != recipe["raw_columns"]:
        raise ValueError("source feature order differs from frozen recipe")
    parts = [raw_features]
    for item in recipe["blocks"]:
        block = get_block(item["block_name"])
        if block.code_sha256() != item["code_sha256"]:
            raise ValueError(f"feature block implementation changed: {block.name}")
        if block.resolve_params(item["params"]) != item["params"]:
            raise ValueError("resolved feature parameters changed")
        if list(block.required_columns(item["params"])) != item["derived_from"]:
            raise ValueError("feature dependencies changed")
        result = block.run(raw_features, item["params"])
        if list(result.columns) != item["columns"]:
            raise ValueError("feature block outputs changed")
        parts.append(result)
    merged = pd.concat(parts, axis=1)
    if merged.columns.duplicated().any():
        raise ValueError("duplicate engineered columns")
    columns = recipe["columns"]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("invalid frozen feature selection")
    return merged[columns]


def export_recipe(feature_set_id, raw_columns, silver_id, feature_root):
    from ml_template.features.feature_set_store import load_feature_set

    if feature_set_id is None:
        return {
            "schema_version": 1,
            "raw_columns": list(raw_columns),
            "columns": list(raw_columns),
            "blocks": [],
        }
    feature_set = load_feature_set(feature_set_id, Path(feature_root))
    if feature_set["silver_dataset_version_id"] != silver_id:
        raise ValueError("feature set belongs to another Silver dataset")
    if set(raw_columns) != set(feature_set["raw_columns"]):
        raise ValueError("raw feature contract changed")
    blocks = []
    for block_id in feature_set["blocks"]:
        item = json.loads((Path(feature_root) / block_id / "manifest.json").read_text())
        if item["silver_dataset_version_id"] != silver_id:
            raise ValueError("feature block belongs to another dataset")
        blocks.append(
            {
                k: item[k]
                for k in [
                    "block_id",
                    "block_name",
                    "code_sha256",
                    "params",
                    "derived_from",
                    "columns",
                ]
            }
        )
    return {
        "schema_version": 1,
        "raw_columns": list(raw_columns),
        "columns": feature_set["columns"],
        "feature_set_id": feature_set_id,
        "blocks": blocks,
    }
