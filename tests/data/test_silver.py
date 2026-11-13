"""Dataset-neutral Silver contract tests.

Uses synthetic frames and temporary output directories; no real Bronze DB.
Requires generic Silver gates. Keeps the existing write_outputs API, including
its fourth, report-only dictionary argument. No AI4I outcome role is supplied.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pandas as pd
import pytest
import yaml

from ml_template.data import silver

N_ROWS = 6
DATASET_NAME = "generic_example"
BRONZE_ID = "example_v1"
ROLES = {
    "id": ["record_id"],
    "feature": ["type", "measurement", "count"],
    "target": ["event"],
}
CONFIG = {
    "data": {
        "entity_id": "Record ID",
        "target": "Event",
        "task": "binary_classification",
        "positive_label": 1,
    },
    "bronze_source_contract": {
        "rows": N_ROWS,
        "roles": {
            "id": ["Record ID"],
            "feature": ["Type", "Measurement", "Count"],
            "target": ["Event"],
        },
    },
    "validation": {
        "required_columns": ["Record ID", "Type", "Measurement", "Count", "Event"],
        "allowed_target_values": [0, 1],
        "require_unique_entity_id": True,
        "max_missing_rate": 0.0,
    },
    "silver_eda_contract": {
        "canonical_names": {
            "Record ID": "record_id",
            "Type": "type",
            "Measurement": "measurement",
            "Count": "count",
            "Event": "event",
        },
        "canonical_dtypes": {
            "record_id": "string",
            "type": "category",
            "measurement": "float64",
            "count": "int64",
            "event": "int8",
        },
        "value_mappings": {
            "event": {
                "False": 0,
                "True": 1,
            },
        },
    },
}


@pytest.fixture
def config():
    return deepcopy(CONFIG)


@pytest.fixture
def roles():
    return deepcopy(ROLES)


@pytest.fixture
def df():
    frame = pd.DataFrame(
        {
            "dataset_version_id": [BRONZE_ID] * N_ROWS,
            "source_row_number": range(1, N_ROWS + 1),
            "record_id": [f"row-{i}" for i in range(N_ROWS)],
            "type": ["standard", "premium"] * 3,
            "measurement": [10.0, 12.5, 8.1, 16.2, 11.0, 9.9],
            "count": [0, 10, 20, 30, 40, 50],
            "event": [0, 1, 0, 1, 1, 0],
        }
    )
    return frame.astype(CONFIG["silver_eda_contract"]["canonical_dtypes"])


def failures(results):
    assert results, "Validation must not return an empty gate list"
    assert all({"check", "passed", "detail"} <= set(r) for r in results)
    return {r["check"] for r in results if not r["passed"]}


def assert_rejected(df, roles, config):
    assert failures(silver.check_gates(df, roles, config))


def test_generic_frame_without_outcome_role_passes(df, roles, config):
    assert "outcome" not in roles
    assert not failures(silver.check_gates(df, roles, config))


def test_non_ai4i_type_values_pass(df, roles, config):
    assert set(df["type"]) == {"standard", "premium"}
    assert not failures(silver.check_gates(df, roles, config))


def test_wrong_row_count_rejected(df, roles, config):
    assert_rejected(df.iloc[:-1], roles, config)


def test_row_count_reads_config(df, roles, config):
    config["bronze_source_contract"]["rows"] = 100
    assert_rejected(df, roles, config)


def test_row_count_is_required(df, roles, config):
    config["bronze_source_contract"].pop("rows")
    with pytest.raises(KeyError, match="rows"):
        silver.check_gates(df, roles, config)


def test_missing_declared_column_rejected(df, roles, config):
    assert_rejected(df.drop(columns="measurement"), roles, config)


def test_numeric_dtype_mismatch_rejected(df, roles, config):
    df["count"] = df["count"].astype("float64")
    assert_rejected(df, roles, config)


def test_configured_key_duplicates_rejected(df, roles, config):
    df.loc[1, "record_id"] = df.loc[0, "record_id"]
    assert_rejected(df, roles, config)


def test_null_key_rejected_even_when_feature_missingness_allowed(df, roles, config):
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    df.loc[0, "record_id"] = pd.NA
    assert_rejected(df, roles, config)


def test_invalid_target_rejected(df, roles, config):
    df.loc[0, "event"] = 2
    assert_rejected(df, roles, config)


def test_null_target_rejected_even_when_feature_missingness_allowed(df, roles, config):
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    df["event"] = df["event"].astype("Int8")
    config["silver_eda_contract"]["canonical_dtypes"]["event"] = "Int8"
    df.loc[0, "event"] = pd.NA
    assert_rejected(df, roles, config)


def test_missing_feature_rejected_by_zero_missingness_policy(df, roles, config):
    df.loc[0, "measurement"] = float("nan")
    assert_rejected(df, roles, config)


def test_existing_feature_missingness_allowed_by_config(df, roles, config):
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    df.loc[0, "measurement"] = float("nan")
    assert not failures(silver.check_gates(df, roles, config))


def test_missingness_over_configured_limit_rejected(df, roles, config):
    # All feature cells missing exceeds either per-column or aggregate limits.
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    df["measurement"] = float("nan")
    df["type"] = pd.Series(pd.Categorical([None] * N_ROWS), index=df.index)
    df["count"] = pd.Series([pd.NA] * N_ROWS, dtype="Int64", index=df.index)
    config["silver_eda_contract"]["canonical_dtypes"]["count"] = "Int64"
    assert_rejected(df, roles, config)


def test_duplicate_source_record_ignores_changed_lineage(df, roles, config):
    duplicate = df.iloc[[0]].copy()
    duplicate["source_row_number"] = 99
    frame = pd.concat([df, duplicate], ignore_index=True)
    config["bronze_source_contract"]["rows"] = len(frame)
    # Isolate duplicate-record detection from unique-key validation.
    config["validation"]["require_unique_entity_id"] = False
    results = silver.check_gates(frame, roles, config)
    assert "no_duplicate_source_records" in failures(results)


def test_validation_does_not_mutate_input(df, roles, config):
    before = df.copy(deep=True)
    silver.check_gates(df, roles, config)
    pd.testing.assert_frame_equal(df, before)


def test_canonicalization_uses_contract_without_real_database(
    df,
    roles,
    config,
    monkeypatch,
):
    reverse = {v: k for k, v in config["silver_eda_contract"]["canonical_names"].items()}
    raw = df.rename(columns=reverse).copy()
    raw["Event"] = raw["Event"].map(
        {
            0: "False",
            1: "True",
        }
    )
    before = raw.copy(deep=True)
    engine = object()

    def extract(*, dataset_version_id, engine):
        assert dataset_version_id == BRONZE_ID
        return raw.copy(deep=True)

    monkeypatch.setattr(silver, "get_engine", lambda: engine)
    monkeypatch.setattr(silver, "extract_bronze", extract)
    actual, actual_roles = silver.load_and_canonicalize(BRONZE_ID, config)
    pd.testing.assert_frame_equal(actual, df)
    assert actual_roles == roles
    pd.testing.assert_frame_equal(raw, before)


def write(df, roles, config, tmp_path, gates=None):
    config_path = tmp_path / "contract.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    if gates is None:
        gates = silver.check_gates(df, roles, config)
    return silver.write_outputs(
        df,
        roles,
        gates,
        dataset_name=DATASET_NAME,
        dataset_version_id=BRONZE_ID,
        config_path=config_path,
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "quality",
    )


def test_passing_build_writes_roundtrip_data_and_verified_manifest(
    df,
    roles,
    config,
    tmp_path,
):
    assert write(df, roles, config, tmp_path) == 0
    versions = list((tmp_path / "silver" / DATASET_NAME).iterdir())
    assert len(versions) == 1
    version = versions[0]
    assert version.name.startswith(f"silver_{BRONZE_ID}_")
    parquet = version / "data.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), df.reset_index(drop=True))
    manifest = json.loads((version / "manifest.json").read_text())
    assert manifest["row_count"] == N_ROWS
    assert manifest["validation_passed"] is True
    assert manifest["roles"] == roles
    assert manifest["parquet_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert (
        manifest["config_sha256"]
        == hashlib.sha256((tmp_path / "contract.yaml").read_bytes()).hexdigest()
    )
    report = tmp_path / "quality" / DATASET_NAME / version.name / "validation.json"
    assert json.loads(report.read_text())["passed"] is True


def test_allowed_missing_value_survives_parquet(df, roles, config, tmp_path):
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    df.loc[0, "measurement"] = float("nan")
    assert write(df, roles, config, tmp_path) == 0
    parquet = next((tmp_path / "silver").rglob("data.parquet"))
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), df.reset_index(drop=True))


def test_failed_gates_write_report_but_no_parquet(df, roles, config, tmp_path):
    gates = [{"check": "synthetic_failure", "passed": False, "detail": "test"}]
    assert write(df, roles, config, tmp_path, gates=gates) == 1
    assert not list(tmp_path.rglob("data.parquet"))
    reports = list((tmp_path / "quality" / DATASET_NAME).glob("*/validation.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["passed"] is False


def test_existing_version_cannot_be_overwritten(df, roles, config, tmp_path):
    assert write(df, roles, config, tmp_path) == 0
    version = next((tmp_path / "silver" / DATASET_NAME).iterdir())
    snapshots = {
        path: path.read_bytes() for path in (version / "data.parquet", version / "manifest.json")
    }
    with pytest.raises(FileExistsError):
        write(df, roles, config, tmp_path)
    for path, before in snapshots.items():
        assert path.read_bytes() == before


def test_changed_contract_has_different_version_id(config, tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = silver.get_silver_version_id(BRONZE_ID, path)
    config["silver_eda_contract"]["missingness"] = {"default_max_rate": 0.25}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert silver.get_silver_version_id(BRONZE_ID, path) != first
