"""Registry tables: one metadata row per Silver build and per EDA run.

Files (Parquet, JSON) stay the source of truth; these rows index them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Column,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
    update,
)

metadata = MetaData()

silver_datasets = Table(
    "silver_datasets",
    metadata,
    Column("silver_dataset_version_id", String, primary_key=True),
    Column("dataset_version_id", String, nullable=False, index=True),
    Column("status", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("row_count", Integer),
    Column("parquet_path", String),
    Column("manifest_path", String),
    Column("validation_path", String),
    Column("config_sha256", String),
    Column("parquet_sha256", String),
    Column("error_summary", Text),
)

eda_runs = Table(
    "eda_runs",
    metadata,
    Column("eda_run_id", String, primary_key=True),
    Column("dataset_version_id", String, nullable=False, index=True),
    Column("status", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("finished_at", String),
    Column("row_count", Integer),
    Column("report_path", String),
    Column("config_sha256", String),
    Column("error_summary", Text),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_tables(engine: Engine) -> None:
    metadata.create_all(engine, checkfirst=True)


def register_silver_build(
    engine: Engine,
    *,
    silver_dataset_version_id: str,
    dataset_version_id: str,
    succeeded: bool,
    validation_path: str,
    manifest_path: str | None = None,
    parquet_path: str | None = None,
    row_count: int | None = None,
    config_sha256: str | None = None,
    parquet_sha256: str | None = None,
    error_summary: str | None = None,
) -> None:
    """Insert one row. A failed row with the same id is replaced on retry."""
    ensure_tables(engine)
    row = {
        "silver_dataset_version_id": silver_dataset_version_id,
        "dataset_version_id": dataset_version_id,
        "status": "succeeded" if succeeded else "failed",
        "created_at": _now(),
        "row_count": row_count,
        "parquet_path": parquet_path,
        "manifest_path": manifest_path,
        "validation_path": validation_path,
        "config_sha256": config_sha256,
        "parquet_sha256": parquet_sha256,
        "error_summary": error_summary,
    }
    with engine.begin() as conn:
        conn.execute(
            delete(silver_datasets).where(
                silver_datasets.c.silver_dataset_version_id == silver_dataset_version_id,
                silver_datasets.c.status == "failed",
            )
        )
        conn.execute(insert(silver_datasets).values(**row))


def latest_valid_silver(
    engine: Engine, dataset_version_id: str | None = None
) -> dict[str, Any] | None:
    """Newest succeeded Silver build, optionally for one Bronze version."""
    ensure_tables(engine)
    stmt = select(silver_datasets).where(silver_datasets.c.status == "succeeded")
    if dataset_version_id is not None:
        stmt = stmt.where(silver_datasets.c.dataset_version_id == dataset_version_id)
    stmt = stmt.order_by(silver_datasets.c.created_at.desc()).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def start_eda_run(engine: Engine, dataset_version_id: str, config_sha256: str | None = None) -> str:
    ensure_tables(engine)
    run_id = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(
            insert(eda_runs).values(
                eda_run_id=run_id,
                dataset_version_id=dataset_version_id,
                status="running",
                started_at=_now(),
                config_sha256=config_sha256,
            )
        )
    return run_id


def finish_eda_run(
    engine: Engine,
    eda_run_id: str,
    *,
    succeeded: bool,
    row_count: int | None = None,
    report_path: str | None = None,
    error_summary: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(eda_runs)
            .where(eda_runs.c.eda_run_id == eda_run_id)
            .values(
                status="succeeded" if succeeded else "failed",
                finished_at=_now(),
                row_count=row_count,
                report_path=report_path,
                error_summary=error_summary,
            )
        )
