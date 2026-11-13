"""Additive SQLite release/batch index. Prediction rows stay in Parquet files."""

import json
from datetime import UTC, datetime

from ml_template.db.evaluation_registry import connection


def now():
    return datetime.now(UTC).isoformat()


def ensure_tables(engine):
    with connection(engine) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_releases (
          release_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL, name TEXT NOT NULL,
          parent_run_id TEXT NOT NULL, calibration_id TEXT, evaluation_id TEXT NOT NULL,
          artifact_dir TEXT NOT NULL, created_at TEXT NOT NULL, policy_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scoring_batches (
          scoring_batch_id TEXT PRIMARY KEY, release_id TEXT NOT NULL, status TEXT NOT NULL,
          started_at TEXT NOT NULL, finished_at TEXT, input_sha256 TEXT, input_name TEXT,
          artifact_dir TEXT NOT NULL, n_rows INTEGER, n_flagged INTEGER, duration_s REAL,
          error_summary TEXT);
        CREATE INDEX IF NOT EXISTS ix_scoring_release ON scoring_batches(release_id);
        """)
        conn.commit()


def register_release(engine, manifest, info, directory):
    ensure_tables(engine)
    with connection(engine) as conn:
        prior = conn.execute(
            "SELECT content_sha256 FROM model_releases WHERE release_id=?",
            (manifest["release_id"],),
        ).fetchone()
        if prior:
            if prior[0] != manifest["content_sha256"]:
                raise ValueError("registered release identity conflict")
            return
        conn.execute(
            "INSERT INTO model_releases VALUES (?,?,?,?,?,?,?,?,?)",
            (
                manifest["release_id"],
                manifest["content_sha256"],
                info["name"],
                info["parent_run_id"],
                info["calibration_id"],
                info["evaluation_id"],
                str(directory),
                info["created_at"],
                json.dumps(info["policy"], allow_nan=False),
            ),
        )
        conn.commit()


def start_batch(engine, batch_id, release_id, directory, input_name):
    ensure_tables(engine)
    with connection(engine) as conn:
        conn.execute(
            "INSERT INTO scoring_batches (scoring_batch_id,release_id,status,started_at,artifact_dir,input_name) VALUES (?,?,'running',?,?,?)",
            (batch_id, release_id, now(), str(directory), input_name),
        )
        conn.commit()


def finish_batch(
    engine, batch_id, *, duration_s, input_sha256=None, n_rows=None, n_flagged=None, error=None
):
    with connection(engine) as conn:
        changed = conn.execute(
            "UPDATE scoring_batches SET status=?,finished_at=?,input_sha256=?,n_rows=?,n_flagged=?,duration_s=?,error_summary=? WHERE scoring_batch_id=? AND status='running'",
            (
                "failed" if error else "succeeded",
                now(),
                input_sha256,
                n_rows,
                n_flagged,
                duration_s,
                str(error)[:2000] if error else None,
                batch_id,
            ),
        )
        if changed.rowcount != 1:
            conn.rollback()
            raise ValueError("only a running scoring execution can be completed")
        conn.commit()
