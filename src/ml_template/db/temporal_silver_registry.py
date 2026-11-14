"A separate immutable multi-file Silver index; does not alter legacy table schemas."

import json
import sqlite3
from pathlib import Path


def register_temporal_silver(database, manifest_path):
    from ml_template.temporal_stream.silver_manifest import verify_inventory

    path = Path(manifest_path).resolve()
    manifest = verify_inventory(path.parent)
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    digest = manifest["manifest_sha256"]
    sid = manifest["silver_dataset_version_id"]
    conn = sqlite3.connect(database, timeout=5)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS temporal_silver_datasets (\n                silver_dataset_version_id TEXT PRIMARY KEY,\n                manifest_sha256 TEXT NOT NULL, manifest_path TEXT NOT NULL,\n                row_count INTEGER NOT NULL, lineage_json TEXT NOT NULL)"
            )
            old = conn.execute(
                "SELECT manifest_sha256,manifest_path FROM temporal_silver_datasets WHERE silver_dataset_version_id=?",
                (sid,),
            ).fetchone()
            if old and old != (digest, str(path)):
                raise ValueError("Immutable temporal Silver registry conflict")
            conn.execute(
                "INSERT OR IGNORE INTO temporal_silver_datasets VALUES (?,?,?,?,?)",
                (
                    sid,
                    digest,
                    str(path),
                    manifest["row_count"],
                    json.dumps(manifest["input"], sort_keys=True),
                ),
            )
    finally:
        conn.close()
    return sid
