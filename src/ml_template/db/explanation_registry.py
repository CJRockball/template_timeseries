"""Additive SQLite execution index; explanation arrays remain in artifact files."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


@contextmanager
def connection(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("existing project SQLite database required")
    conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def start(path, explanation_id, config, binding, directory):
    with connection(path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS explanation_runs (
          explanation_id TEXT PRIMARY KEY,
          release_id TEXT NOT NULL,
          scoring_batch_id TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed')),
          started_at TEXT NOT NULL, finished_at TEXT,
          artifact_dir TEXT NOT NULL, content_sha256 TEXT,
          config_json TEXT NOT NULL, binding_sha256 TEXT NOT NULL,
          n_rows INTEGER, n_features INTEGER, duration_s REAL, error_summary TEXT);
        CREATE INDEX IF NOT EXISTS ix_explanation_release_batch
          ON explanation_runs(release_id,scoring_batch_id);
        """)
        conn.execute(
            "INSERT INTO explanation_runs "
            "(explanation_id,release_id,scoring_batch_id,status,started_at,artifact_dir,config_json,binding_sha256) "
            "VALUES (?,?,?,'running',?,?,?,?)",
            (
                explanation_id,
                binding["release_id"],
                binding["scoring_batch_id"],
                now(),
                str(directory),
                json.dumps(config, sort_keys=True, allow_nan=False),
                binding["binding_sha256"],
            ),
        )


def finish(
    path,
    explanation_id,
    *,
    duration_s,
    manifest=None,
    n_rows=None,
    n_features=None,
    error=None,
    artifact_dir=None,
):
    with connection(path) as conn:
        cur = conn.execute(
            "UPDATE explanation_runs SET status=?,finished_at=?,duration_s=?,content_sha256=?,"
            "n_rows=?,n_features=?,error_summary=?,artifact_dir=COALESCE(?,artifact_dir) WHERE explanation_id=? AND status='running'",
            (
                "failed" if error is not None else "succeeded",
                now(),
                duration_s,
                None if manifest is None else manifest["content_sha256"],
                n_rows,
                n_features,
                None if error is None else str(error)[:2000],
                None if artifact_dir is None else str(artifact_dir),
                explanation_id,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError("only a running explanation execution may be completed")
