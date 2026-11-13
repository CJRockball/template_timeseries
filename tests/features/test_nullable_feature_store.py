"""Run in the full repository: Parquet persistence and frozen-recipe export."""

import pandas as pd
from pandas.testing import assert_frame_equal

from ml_template.features.block_registry import get_block
from ml_template.features.blocks import generic_interactions  # noqa: F401
from ml_template.features.feature_recipe import export_recipe, replay_recipe
from ml_template.features.feature_set_store import (
    build_block,
    create_feature_set,
    load_block,
    resolve_feature_set,
)


def example():
    data = pd.DataFrame(
        {
            "x": pd.array([2, None, 8], dtype="Int64"),
            "y": pd.array([1, 2, None], dtype="Int64"),
        },
        index=pd.Index([10, 30, 20], name="id"),
    )
    params = {
        "numeric_pairs": [{"a": "x", "b": "y", "ops": ["difference"]}],
        "nullable_outputs": ["x__minus__y"],
    }
    return data, params


def test_nullable_block_parquet_roundtrip_and_reuse(tmp_path):
    data, params = example()
    manifest = build_block("generic_interactions", data, "silver_test", params, tmp_path)
    saved, result = load_block(manifest["block_id"], tmp_path)
    expected = get_block("generic_interactions").run(data, params)
    assert_frame_equal(result, expected)
    assert saved["params"]["nullable_outputs"] == params["nullable_outputs"]
    again = build_block("generic_interactions", data, "silver_test", params, tmp_path)
    assert again == manifest


def test_nullable_feature_set_export_and_unseen_row_replay(tmp_path):
    data, params = example()
    manifest = build_block("generic_interactions", data, "silver_test", params, tmp_path)
    feature_set = create_feature_set(
        "silver_test",
        [manifest["block_id"]],
        raw_columns=list(data.columns),
        root=tmp_path,
    )
    recipe = export_recipe(
        feature_set["feature_set_id"], list(data.columns), "silver_test", tmp_path
    )
    materialized = resolve_feature_set(feature_set["feature_set_id"], data, "silver_test", tmp_path)
    assert_frame_equal(replay_recipe(data, recipe), materialized)
    unseen = data.copy()
    unseen.index = pd.Index([101, 102, 103], name="id")
    result = replay_recipe(unseen, recipe)
    assert result.index.equals(unseen.index)
    assert result["x__minus__y"].isna().tolist() == [False, True, True]
