"""Tests for source inspection, hashing, contract checks, and chunk streaming."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_template.data.inspect_bronze import (
    compute_schema_hash,
    inspect_source_schema,
    iter_batches,
    validate_bronze_contract,
)
from ml_template.paths import ROOT


@pytest.fixture
def ai4i_fixture_path() -> Path:
    path = ROOT / "tests" / "fixtures" / "ai4i_small.csv"
    if not path.is_file():
        pytest.skip(f"Fixture missing: {path}")
    return path


@pytest.fixture
def ai4i_contract() -> dict:
    return {
        "columns": 14,
        "col_names": [
            "UDI",
            "Product ID",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
        ],
    }


def test_inspect_source_schema_ai4i(ai4i_fixture_path: Path) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    assert schema.format == "csv"
    assert len(schema.columns) == 14
    assert "UDI" in schema.columns
    assert "Torque [Nm]" in schema.columns
    assert "Machine failure" in schema.columns
    assert len(schema.source_sha256) == 64
    assert len(schema.schema_hash) == 16


def test_inspect_source_schema_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_source_schema(tmp_path / "non_existent.csv")


def test_compute_schema_hash_deterministic() -> None:
    cols = ["col_a", "col_b"]
    types = {"col_a": "string", "col_b": "int64"}
    h1 = compute_schema_hash(cols, types)
    h2 = compute_schema_hash(cols, types)
    assert h1 == h2
    assert h1 != compute_schema_hash(cols[::-1], types)


def test_validate_bronze_contract_success(ai4i_fixture_path: Path, ai4i_contract: dict) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    validate_bronze_contract(schema, ai4i_contract)


def test_validate_bronze_contract_column_count_mismatch(ai4i_fixture_path: Path) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    bad_contract = {"columns": 999}
    with pytest.raises(AssertionError, match="expected 999 columns"):
        validate_bronze_contract(schema, bad_contract)


def test_validate_bronze_contract_missing_columns(ai4i_fixture_path: Path) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    bad_contract = {"col_names": ["UDI", "NonExistentColumn"]}
    with pytest.raises(AssertionError, match="missing required column"):
        validate_bronze_contract(schema, bad_contract)


def test_iter_batches_streams_rows(ai4i_fixture_path: Path) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    batches = list(iter_batches(schema, batch_size=10))

    total_rows = sum(len(b) for b in batches)
    raw_df = pd.read_csv(ai4i_fixture_path)
    assert total_rows == len(raw_df)
    assert len(batches) >= 1
