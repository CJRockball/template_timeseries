"""Tests for dynamic SQLite table creation and schema definitions."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect

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


def test_setup_db_creates_expected_tables(sqlite_engine, ai4i_fixture_path: Path) -> None:
    schema = inspect_source_schema(ai4i_fixture_path)
    setup_db(sqlite_engine, schema)

    inspector = sa_inspect(sqlite_engine)
    tables = inspector.get_table_names()

    assert "raw_records" in tables
    assert "column_registry" in tables
    assert "ingestion_log" in tables

    raw_cols = [c["name"] for c in inspector.get_columns("raw_records")]
    assert "dataset_version_id" in raw_cols
    assert "source_row_number" in raw_cols
    assert "Air temperature [K]" in raw_cols
    assert "Machine failure" in raw_cols
