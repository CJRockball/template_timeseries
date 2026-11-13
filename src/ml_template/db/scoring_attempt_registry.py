"""Additive attempt ledger; prediction and monitoring records remain separate."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime


@contextmanager
def connection(engine):
    owned = not isinstance(engine, sqlite3.Connection)
    if owned and engine.dialect.name != "sqlite":
        raise ValueError("scoring attempts support SQLite only")
    conn = engine.raw_connection() if owned else engine
    try:
        yield conn
    finally:
        if owned:
            conn.close()


def now():
    return datetime.now(UTC).isoformat()


def ensure_tables(engine):
    with connection(engine) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS scoring_attempts (
          attempt_id TEXT PRIMARY KEY,status TEXT NOT NULL,stage TEXT NOT NULL,
          started_at TEXT NOT NULL,finished_at TEXT,scoring_batch_id TEXT,
          requested_release_dir TEXT NOT NULL,input_name TEXT NOT NULL,metadata_json TEXT NOT NULL,
          duration_s REAL,timings_json TEXT,error_type TEXT,error_summary TEXT,validation_json TEXT);
        CREATE INDEX IF NOT EXISTS ix_scoring_attempt_batch ON scoring_attempts(scoring_batch_id);
        """)
        conn.commit()


def start(engine, identity, release_dir, input_name):
    ensure_tables(engine)
    with connection(engine) as conn:
        conn.execute(
            """INSERT INTO scoring_attempts
        (attempt_id,status,stage,started_at,requested_release_dir,input_name,metadata_json)
        VALUES (?,'running','metadata_validation',?,?,?,'{}')""",
            (identity, now(), str(release_dir), input_name),
        )
        conn.commit()


def metadata(engine, identity, value):
    with connection(engine) as conn:
        cursor = conn.execute(
            "UPDATE scoring_attempts SET metadata_json=? WHERE attempt_id=? AND status='running'",
            (json.dumps(value, allow_nan=False), identity),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("attempt must be running")
        conn.commit()


def stage(engine, identity, value, batch_id=None):
    with connection(engine) as conn:
        cursor = conn.execute(
            "UPDATE scoring_attempts SET stage=?,scoring_batch_id=COALESCE(?,scoring_batch_id) WHERE attempt_id=? AND status='running'",
            (value, batch_id, identity),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("attempt must be running")
        conn.commit()


def finish(engine, identity, duration_s, timings, *, batch_id=None, error=None):
    with connection(engine) as conn:
        row = conn.execute(
            "SELECT stage FROM scoring_attempts WHERE attempt_id=? AND status='running'",
            (identity,),
        ).fetchone()
        if row is None:
            raise ValueError("completed attempts are immutable")
        report = (
            None
            if error is None
            else {
                "accepted": False,
                "stage": row[0],
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
                "scope": "first raised exception, not exhaustive row-level validation",
            }
        )
        cursor = conn.execute(
            """UPDATE scoring_attempts SET status=?,finished_at=?,duration_s=?,timings_json=?,
        scoring_batch_id=COALESCE(?,scoring_batch_id),error_type=?,error_summary=?,validation_json=?
        WHERE attempt_id=? AND status='running'""",
            (
                "failed" if error is not None else "succeeded",
                now(),
                duration_s,
                json.dumps(timings, allow_nan=False),
                batch_id,
                type(error).__name__ if error is not None else None,
                str(error)[:2000] if error is not None else None,
                json.dumps(report, allow_nan=False) if report else None,
                identity,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("completed attempts are immutable")
        conn.commit()
