"""Tests for column registry construction, atomic batch insertion, and lineage tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from ml_template.data.ingest import _to_canonical_name, build_registry, insert_batches
from ml_template.data.inspect_bronze import inspect_source_schema
from ml_template.db.bronze_tables import setup_db
from ml_template.paths import ROOT


@pytest.fixture
def sqlite_engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def ai4i_fixture_path() -> Path:
    path = ROOT / "tests" / "fixtures" / "ai4i_small.csv"
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")
    return path


@pytest.fixture
def ai4i_contract() -> dict:
    return {
        "roles": {
            "id": ["UDI", "Product ID"],
            "features": [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ],
            "target": ["Machine failure"],
            "outcomes": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        },
    }


def test_to_canonical_name() -> None:
    assert _to_canonical_name("Air temperature [K]") == "air_temperature_k"
    assert _to_canonical_name("Rotational speed [rpm]") == "rotational_speed_rpm"
    assert _to_canonical_name("Product ID") == "product_id"


def test_build_registry(ai4i_fixture_path: Path, ai4i_contract: dict) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    registry_df = build_registry(schema, "ds_v1", ai4i_contract)

    assert len(registry_df) == 14
    assert set(registry_df["dataset_version_id"]) == {"ds_v1"}

    udi_row = registry_df[registry_df["column_name"] == "UDI"].iloc[0]
    assert udi_row["semantic_role"] == "id"
    assert udi_row["canonical_column_name"] == "udi"

    target_row = registry_df[registry_df["column_name"] == "Machine failure"].iloc[0]
    assert target_row["semantic_role"] == "target"
    assert target_row["canonical_column_name"] == "machine_failure"


def test_insert_batches_roundtrip(
    sqlite_engine, ai4i_fixture_path: Path, ai4i_contract: dict
) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    setup_db(sqlite_engine, schema)

    registry = build_registry(schema, "ds_test_v1", ai4i_contract)
    started_at = datetime.now(UTC)
    expected_rows = len(pd.read_csv(ai4i_fixture_path))

    rows_ingested = insert_batches(
        engine=sqlite_engine,
        schema=schema,
        registry=registry,
        dataset_version_id="ds_test_v1",
        git_sha="test_git_sha",
        started_at=started_at,
        batch_size=10,
    )

    assert rows_ingested == expected_rows

    with sqlite_engine.connect() as conn:
        res_raw = conn.execute(text("SELECT * FROM raw_records")).mappings().all()
        assert len(res_raw) == expected_rows
        assert res_raw[0]["dataset_version_id"] == "ds_test_v1"
        assert res_raw[0]["source_row_number"] == 1
        assert res_raw[-1]["source_row_number"] == expected_rows

        res_reg = conn.execute(text("SELECT * FROM column_registry")).mappings().all()
        assert len(res_reg) == 14

        res_log = conn.execute(text("SELECT * FROM ingestion_log")).mappings().all()
        assert len(res_log) == 1
        assert res_log[0]["status"] == "succeeded"
        assert res_log[0]["row_count_written"] == expected_rows
        assert res_log[0]["git_sha"] == "test_git_sha"


def test_insert_batches_idempotent_overwrite(
    sqlite_engine, ai4i_fixture_path: Path, ai4i_contract: dict
) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    setup_db(sqlite_engine, schema)
    registry = build_registry(schema, "ds_v1", ai4i_contract)
    started_at = datetime.now(UTC)
    expected_rows = len(pd.read_csv(ai4i_fixture_path))

    # Ingesting the same dataset version twice must replace, not duplicate
    insert_batches(sqlite_engine, schema, registry, "ds_v1", "sha", started_at)
    insert_batches(sqlite_engine, schema, registry, "ds_v1", "sha", started_at)

    with sqlite_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM raw_records")).scalar()
        assert count == expected_rows
