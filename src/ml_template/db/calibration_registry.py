"""Additive calibration tables; never write to the parent's metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, insert, select, update

metadata = MetaData()
calibration_runs = Table(
    "calibration_runs",
    metadata,
    Column("calibration_id", String, primary_key=True),
    Column("parent_run_id", String, nullable=False, index=True),
    Column("status", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("finished_at", String),
    Column("method", String, nullable=False),
    Column("config_json", Text, nullable=False),
    Column("artifact_dir", Text, nullable=False),
    Column("error_summary", Text),
)
calibration_metrics = Table(
    "calibration_metrics",
    metadata,
    Column("calibration_id", String, primary_key=True),
    Column("split", String, primary_key=True),
    Column("variant", String, primary_key=True),
    Column("fold", Integer, primary_key=True),
    Column("metric", String, primary_key=True),
    Column("value", Float),
)


def start_calibration(engine, calibration_id, parent_run_id, method, config, artifact_dir):
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(calibration_runs).values(
                calibration_id=calibration_id,
                parent_run_id=parent_run_id,
                status="running",
                started_at=datetime.now(UTC).isoformat(),
                method=method,
                config_json=json.dumps(config, sort_keys=True),
                artifact_dir=artifact_dir,
            )
        )


def finish_calibration(engine, calibration_id, *, succeeded, rows=(), error=None):
    with engine.begin() as conn:
        result = conn.execute(
            update(calibration_runs)
            .where(calibration_runs.c.calibration_id == calibration_id)
            .values(
                status="succeeded" if succeeded else "failed",
                finished_at=datetime.now(UTC).isoformat(),
                error_summary=error,
            )
        )
        if result.rowcount != 1:
            raise KeyError(calibration_id)
        if rows:
            conn.execute(
                insert(calibration_metrics),
                [dict(calibration_id=calibration_id, **r) for r in rows],
            )


def get_calibration(engine, calibration_id):
    metadata.create_all(engine)
    with engine.connect() as conn:
        row = (
            conn.execute(
                select(calibration_runs).where(calibration_runs.c.calibration_id == calibration_id)
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None
