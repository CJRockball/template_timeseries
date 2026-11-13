"""Operational references, label versions and immutable monitoring executions."""

import json

from ml_template.db.evaluation_registry import connection
from ml_template.monitoring.artifacts import now


def ensure_tables(engine):
    with connection(engine) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS monitoring_objects (
          object_id TEXT PRIMARY KEY, kind TEXT NOT NULL, scoring_batch_id TEXT NOT NULL,
          release_id TEXT NOT NULL, artifact_dir TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
          created_at TEXT NOT NULL, metadata_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_monitoring_object_batch ON monitoring_objects(scoring_batch_id, kind);
        CREATE TABLE IF NOT EXISTS monitoring_runs (
          monitoring_id TEXT PRIMARY KEY, scoring_batch_id TEXT NOT NULL, release_id TEXT,
          reference_id TEXT NOT NULL, label_import_id TEXT, status TEXT NOT NULL,
          started_at TEXT NOT NULL, finished_at TEXT, artifact_dir TEXT NOT NULL,
          manifest_sha256 TEXT, assessment TEXT, config_json TEXT NOT NULL, error_summary TEXT);
        CREATE INDEX IF NOT EXISTS ix_monitoring_run_batch ON monitoring_runs(scoring_batch_id, started_at);
        """)
        conn.commit()


def rows(engine, sql, params=()):
    with connection(engine) as conn:
        cursor = conn.execute(sql, params)
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def get_object(engine, object_id, kind):
    found = rows(
        engine, "SELECT * FROM monitoring_objects WHERE object_id=? AND kind=?", (object_id, kind)
    )
    if len(found) != 1:
        raise ValueError(f"unknown {kind}: {object_id}")
    return found[0]


def register_object(
    engine, object_id, kind, batch_id, release_id, directory, manifest_hash, metadata
):
    ensure_tables(engine)
    with connection(engine) as conn:
        conn.execute(
            """INSERT INTO monitoring_objects
        (object_id,kind,scoring_batch_id,release_id,artifact_dir,manifest_sha256,created_at,metadata_json)
        VALUES (?,?,?,?,?,?,?,?)""",
            (
                object_id,
                kind,
                batch_id,
                release_id,
                str(directory),
                manifest_hash,
                now(),
                json.dumps(metadata, allow_nan=False),
            ),
        )
        conn.commit()


def start_run(engine, identity, config, directory):
    ensure_tables(engine)
    with connection(engine) as conn:
        conn.execute(
            """INSERT INTO monitoring_runs
        (monitoring_id,scoring_batch_id,reference_id,label_import_id,status,started_at,artifact_dir,config_json)
        VALUES (?,?,?,?,'running',?,?,?)""",
            (
                identity,
                config.scoring_batch_id,
                config.reference_id,
                config.label_import_id,
                now(),
                str(directory),
                config.model_dump_json(),
            ),
        )
        conn.commit()


def finish_run(
    engine, identity, *, release_id=None, manifest_hash=None, assessment=None, error=None
):
    with connection(engine) as conn:
        changed = conn.execute(
            """UPDATE monitoring_runs SET status=?,finished_at=?,release_id=?,
        manifest_sha256=?,assessment=?,error_summary=? WHERE monitoring_id=? AND status='running'""",
            (
                "failed" if error else "succeeded",
                now(),
                release_id,
                manifest_hash,
                assessment,
                str(error)[:2000] if error else None,
                identity,
            ),
        )
        if changed.rowcount != 1:
            conn.rollback()
            raise ValueError("only a running monitoring execution can finish")
        conn.commit()
