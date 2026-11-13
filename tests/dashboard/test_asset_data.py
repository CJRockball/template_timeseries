import json

import pandas as pd
import pytest

from ml_template.dashboard import asset_data as ad
from ml_template.dashboard.inventory_common import file_state, json_list, singapore_times


def _features():
    sets = pd.DataFrame(
        [
            {
                "feature_set_id": "fs1",
                "parent_feature_set_id": None,
                "silver_dataset_version_id": "s",
                "blocks_json": '["thermal"]',
                "columns_json": '["speed", "power"]',
                "n_features": 2,
                "description": "base",
            },
            {
                "feature_set_id": "fs2",
                "parent_feature_set_id": "fs1",
                "silver_dataset_version_id": "s",
                "blocks_json": '["thermal"]',
                "columns_json": '["power"]',
                "n_features": 1,
                "description": "selection",
            },
        ]
    )
    blocks = pd.DataFrame([{"block_id": "thermal", "columns_json": '["power"]'}])
    return sets, blocks


def test_column_order_and_sources():
    sets, blocks = _features()
    table = ad.feature_columns(sets, blocks, "fs1")
    assert table.feature.tolist() == ["speed", "power"]
    assert table.source.tolist() == ["Silver", "thermal"]
    assert table.position.tolist() == [1, 2]


def test_selected_feature_set():
    sets, blocks = _features()
    assert ad.feature_columns(sets, blocks, "fs2").feature.tolist() == ["power"]
    assert ad.feature_columns(sets, blocks, "missing").empty


def test_missing_block_not_silently_called_raw():
    sets, blocks = _features()
    table = ad.feature_columns(sets, blocks.iloc[:0], "fs1")
    assert set(table.source) == {"unresolved"}


def test_lineage_status():
    sets, _ = _features()
    assert ad.lineage_table(sets).parent_status.tolist() == ["root", "registered"]
    sets.loc[1, "parent_feature_set_id"] = "missing"
    assert ad.lineage_table(sets).iloc[1].parent_status == "missing registry row"


def test_manifest_metadata_only(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"row_count": 100, "roles": {"target": ["failure"]}, "validation_passed": True})
    )
    result = ad.dataset_details({"manifest_path": "manifest.json"}, tmp_path)
    assert result["row_count"] == 100
    assert result["roles"] == {"target": ["failure"]}
    assert "message" in ad.dataset_details({}, tmp_path)


def test_file_presence_is_not_hash_validation(tmp_path):
    (tmp_path / "file").write_text("x")
    assert file_state("file", tmp_path) == "present"
    assert file_state("absent", tmp_path) == "missing or unrecorded"


def test_model_ranking_cohort_and_direction():
    rows = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "experiment_name": "e",
            "silver_dataset_version_id": "s",
            "split_id": "sp",
            "primary_metric": "pr_auc",
            "primary_value": [0.7, 0.8, 0.99],
            "status": ["succeeded", "succeeded", "failed"],
        }
    )
    assert ad.ranked_models(rows).run_id.tolist() == ["b", "a"]
    assert ad.ranked_models(rows, higher=False).run_id.tolist() == ["a", "b"]
    rows.loc[0, "split_id"] = "other"
    with pytest.raises(ValueError, match="Select one"):
        ad.ranked_models(rows)


def test_json_list_validation():
    assert json_list(None) == []
    assert json_list('["a"]') == ["a"]
    with pytest.raises(ValueError):
        json_list('{"a": 1}')


def test_singapore_dates():
    table = singapore_times(pd.DataFrame({"created_at": ["2026-10-16T01:00:00Z", None]}))
    assert table.iloc[0]["created_at_sgt"] == "2026-10-16 09:00:00"
    assert pd.isna(table.iloc[1]["created_at_sgt"])
