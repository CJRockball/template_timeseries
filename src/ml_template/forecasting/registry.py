"Separate SQLite index for immutable temporal plans."

import sqlite3
from contextlib import closing
from pathlib import Path

from .artifacts import verify_plan


def register_plan(database, plan_path):
    path = Path(plan_path).resolve()
    manifest = verify_plan(path)
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError("Existing project SQLite database required")
    with closing(sqlite3.connect(database, timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS temporal_origin_plans (\n            plan_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL,\n            path TEXT NOT NULL, silver_id TEXT NOT NULL, request_count INTEGER NOT NULL)"
        )
        row = (
            manifest["plan_id"],
            manifest["manifest_sha256"],
            str(path),
            manifest["identity"]["silver"]["silver_dataset_version_id"],
            manifest["validation"]["requests"],
        )
        old = conn.execute(
            "SELECT plan_id,manifest_sha256,path,silver_id,request_count FROM temporal_origin_plans WHERE plan_id=?",
            (row[0],),
        ).fetchone()
        if old is not None and old != row:
            raise ValueError("Immutable origin registry conflict")
        conn.execute("INSERT OR IGNORE INTO temporal_origin_plans VALUES (?,?,?,?,?)", row)
    return manifest["plan_id"]
