from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select

from ml_template.db import data_run_registry


@pytest.fixture
def engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def _silver(engine, version, bronze="ds_v1", ok=True):
    data_run_registry.register_silver_build(
        engine,
        silver_dataset_version_id=version,
        dataset_version_id=bronze,
        succeeded=ok,
        validation_path="v.json",
        manifest_path="m.json" if ok else None,
        parquet_path="d.parquet" if ok else None,
        row_count=10_000 if ok else None,
    )


def test_latest_valid_silver_ignores_failed_builds(engine):
    _silver(engine, "silver_a", ok=True)
    _silver(engine, "silver_b", ok=False)
    latest = data_run_registry.latest_valid_silver(engine, "ds_v1")
    assert latest is not None
    assert latest["silver_dataset_version_id"] == "silver_a"
    assert latest["row_count"] == 10_000


def test_latest_valid_silver_none_when_empty(engine):
    assert data_run_registry.latest_valid_silver(engine, "ds_v1") is None


def test_failed_build_can_be_retried_with_same_id(engine):
    _silver(engine, "silver_a", ok=False)
    _silver(engine, "silver_a", ok=True)
    with engine.connect() as conn:
        rows = conn.execute(select(data_run_registry.silver_datasets)).mappings().all()
    assert len(rows) == 1
    assert rows[0]["status"] == "succeeded"


def test_eda_run_lifecycle(engine):
    run_id = data_run_registry.start_eda_run(engine, "ds_v1", config_sha256="abc")
    data_run_registry.finish_eda_run(
        engine, run_id, succeeded=True, row_count=27, report_path="r.json"
    )
    with engine.connect() as conn:
        row = conn.execute(select(data_run_registry.eda_runs)).mappings().one()
    assert row["status"] == "succeeded"
    assert row["row_count"] == 27
    assert row["finished_at"] is not None


def test_eda_run_failure_is_recorded(engine):
    run_id = data_run_registry.start_eda_run(engine, "ds_v1")
    data_run_registry.finish_eda_run(engine, run_id, succeeded=False, error_summary="boom")
    with engine.connect() as conn:
        row = conn.execute(select(data_run_registry.eda_runs)).mappings().one()
    assert row["status"] == "failed"
    assert row["error_summary"] == "boom"
