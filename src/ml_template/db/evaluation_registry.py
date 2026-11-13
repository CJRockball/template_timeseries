"""Additive SQLite evaluation registry; accepts sqlite3 or SQLAlchemy SQLite Engine."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime


@contextmanager
def connection(engine):
    owned = not isinstance(engine, sqlite3.Connection)
    if owned and engine.dialect.name != "sqlite":
        raise ValueError("evaluation registry supports SQLite only")
    c = engine.raw_connection() if owned else engine
    try:
        yield c
    finally:
        if owned:
            c.close()


def now():
    return datetime.now(UTC).isoformat()


def ensure_tables(engine):
    with connection(engine) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS final_evaluations (
          evaluation_id TEXT PRIMARY KEY, identity_hash TEXT NOT NULL, active_identity TEXT UNIQUE,
          parent_run_id TEXT NOT NULL, calibration_id TEXT, silver_id TEXT NOT NULL,
          holdout_fingerprint TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
          finished_at TEXT, accessed_at TEXT, artifact_dir TEXT NOT NULL, spec_json TEXT NOT NULL,
          prior_json TEXT NOT NULL, supersedes_evaluation_id TEXT, error_summary TEXT);
        CREATE TABLE IF NOT EXISTS final_evaluation_metrics (
          evaluation_id TEXT NOT NULL, variant TEXT NOT NULL, metric TEXT NOT NULL,
          result_json TEXT NOT NULL, PRIMARY KEY(evaluation_id, variant, metric));
        CREATE TABLE IF NOT EXISTS final_evaluation_locks (silver_id TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL);
        """)
        c.commit()


def query(engine, sql, params=()):
    ensure_tables(engine)
    with connection(engine) as c:
        cursor = c.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def get_evaluation(engine, evaluation_id):
    rows = query(engine, "SELECT * FROM final_evaluations WHERE evaluation_id=?", (evaluation_id,))
    return rows[0] if rows else None


def cached_evaluation(engine, identity_hash):
    rows = query(
        engine,
        "SELECT * FROM final_evaluations WHERE active_identity=? AND status='succeeded'",
        (identity_hash,),
    )
    return rows[0] if rows else None


def prior_exposures(engine, silver_id):
    return query(
        engine,
        "SELECT evaluation_id, holdout_fingerprint, accessed_at FROM final_evaluations WHERE silver_id=? AND accessed_at IS NOT NULL ORDER BY started_at,evaluation_id",
        (silver_id,),
    )


def start_evaluation(
    engine, *, evaluation_id, identity_hash, spec, fingerprint, artifact_dir, prior
):
    ensure_tables(engine)
    with connection(engine) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute(
                "INSERT INTO final_evaluation_locks VALUES (?,?)",
                (spec.expected_silver_dataset_version_id, evaluation_id),
            )
            c.execute(
                """INSERT INTO final_evaluations (evaluation_id, identity_hash, active_identity,
                parent_run_id, calibration_id, silver_id, holdout_fingerprint, status, started_at,
                artifact_dir, spec_json, prior_json, supersedes_evaluation_id)
                VALUES (?,?,?,?,?,?,?,'running',?,?,?,?,?)""",
                (
                    evaluation_id,
                    identity_hash,
                    identity_hash,
                    spec.parent_run_id,
                    spec.calibration_id,
                    spec.expected_silver_dataset_version_id,
                    fingerprint,
                    now(),
                    str(artifact_dir),
                    spec.model_dump_json(),
                    json.dumps(prior),
                    spec.supersedes_evaluation_id,
                ),
            )
            c.commit()
        except sqlite3.IntegrityError as exc:
            c.rollback()
            raise RuntimeError("identity exists or dataset evaluation lock is held") from exc
        except Exception:
            c.rollback()
            raise


def mark_accessed(engine, evaluation_id, parent_run_id):
    with connection(engine) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            stamp = now()
            cursor = c.execute(
                "UPDATE final_evaluations SET accessed_at=? WHERE evaluation_id=? AND status='running' AND accessed_at IS NULL",
                (stamp, evaluation_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("evaluation must be running and not previously accessed")
            c.execute(
                "UPDATE model_runs SET holdout_evaluated_at=? WHERE run_id=? AND holdout_evaluated_at IS NULL",
                (stamp, parent_run_id),
            )
            c.commit()
        except Exception:
            c.rollback()
            raise


def finish_evaluation(engine, evaluation_id, *, rows=(), error=None):
    with connection(engine) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT status FROM final_evaluations WHERE evaluation_id=?", (evaluation_id,)
            ).fetchone()
            if not row or row[0] != "running":
                raise ValueError("completed evaluations are immutable")
            if error is None:
                c.executemany(
                    "INSERT INTO final_evaluation_metrics VALUES (?,?,?,?)",
                    [
                        (evaluation_id, r["variant"], r["metric"], json.dumps(r, allow_nan=False))
                        for r in rows
                    ],
                )
                c.execute(
                    "UPDATE final_evaluations SET status='succeeded',finished_at=? WHERE evaluation_id=?",
                    (now(), evaluation_id),
                )
            else:
                c.execute(
                    "UPDATE final_evaluations SET status='failed',active_identity=NULL,finished_at=?,error_summary=? WHERE evaluation_id=?",
                    (now(), str(error)[:2000], evaluation_id),
                )
            c.execute("DELETE FROM final_evaluation_locks WHERE evaluation_id=?", (evaluation_id,))
            c.commit()
        except Exception:
            c.rollback()
            raise
