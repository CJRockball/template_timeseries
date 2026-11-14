"""Immutable, schema-independent SQLite source snapshots.

Uses dedicated tables; never touches legacy raw_records or Silver registries.
"""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .contracts import SourceContract, canonical_json


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def frame_hash(frame):
    rows = [
        [None if pd.isna(v) else v for v in row] for row in frame.itertuples(index=False, name=None)
    ]
    return digest_bytes(canonical_json({"columns": list(frame.columns), "rows": rows}).encode())


@contextmanager
def connection(db):
    path = Path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


def setup(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS temporal_sources (
        source_id TEXT NOT NULL, version TEXT NOT NULL,
        table_name TEXT NOT NULL UNIQUE, manifest_json TEXT NOT NULL,
        PRIMARY KEY(source_id,version))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS temporal_assemblies (
        assembly_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL)""")


def safe_name(name):
    if not re.fullmatch(r"ts_[0-9a-f]{64}", name):
        raise ValueError("Invalid internal table identifier")
    return '"' + name + '"'


def ingest_csv(db, source_id, version, path, contract, provenance=None):
    if not source_id or not version:
        raise ValueError("Source ID and explicit version required")
    contract = SourceContract.from_dict(contract) if isinstance(contract, dict) else contract
    payload = Path(path).read_bytes()
    # Parse the exact bytes being fingerprinted, not a second read of a mutable file.
    from io import BytesIO

    frame = pd.read_csv(
        BytesIO(payload), dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig"
    )
    frame = contract.validate(frame)
    fingerprint = digest_bytes(
        canonical_json(
            {
                "source_sha256": digest_bytes(payload),
                "contract": contract.to_dict(),
                "provenance": provenance or {},
            }
        ).encode()
    )
    table = "ts_" + digest_bytes(canonical_json([source_id, version]).encode())
    manifest = {
        "source_id": source_id,
        "version": version,
        "fingerprint": fingerprint,
        "source_sha256": digest_bytes(payload),
        "contract": contract.to_dict(),
        "provenance": provenance or {},
        "rows": len(frame),
        "data_sha256": frame_hash(frame),
        "created_at": datetime.now(UTC).isoformat(),
    }
    with connection(db) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        setup(conn)
        prior = conn.execute(
            "SELECT manifest_json FROM temporal_sources WHERE source_id=? AND version=?",
            (source_id, version),
        ).fetchone()
        if prior:
            old = json.loads(prior[0])
            if old["fingerprint"] != fingerprint:
                raise ValueError("Immutable source/version conflict; choose a new version")
            existing = _read(conn, source_id, version)
            if frame_hash(existing) != old["data_sha256"]:
                raise ValueError("Existing source data integrity failure")
            return {**old, "reused": True}
        sql_types = {"int": "INTEGER", "float": "REAL"}

        # User column identifiers are safely quoted, never interpolated unescaped.
        def quote(identifier: str) -> str:
            return '"' + identifier.replace('"', '""') + '"'

        columns = [
            quote(c)
            + " "
            + sql_types.get(k, "TEXT")
            + ("" if c in contract.nullable else " NOT NULL")
            for c, k in sorted(contract.columns.items())
        ]
        columns.append("UNIQUE (" + ",".join(quote(c) for c in contract.key) + ")")
        conn.execute("CREATE TABLE " + safe_name(table) + " (" + ",".join(columns) + ")")
        records = [
            [None if pd.isna(v) else v for v in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        conn.executemany(
            "INSERT INTO "
            + safe_name(table)
            + " VALUES ("
            + ",".join("?" for _ in frame.columns)
            + ")",
            records,
        )
        conn.execute(
            "INSERT INTO temporal_sources VALUES (?,?,?,?)",
            (source_id, version, table, canonical_json(manifest)),
        )
        if frame_hash(_read(conn, source_id, version)) != manifest["data_sha256"]:
            raise ValueError("SQLite round-trip integrity failure")
    return {**manifest, "reused": False}


def _read(conn, source_id, version):
    row = conn.execute(
        "SELECT table_name,manifest_json FROM temporal_sources WHERE source_id=? AND version=?",
        (source_id, version),
    ).fetchone()
    if not row:
        raise KeyError((source_id, version))
    spec = json.loads(row[1])
    cursor = conn.execute("SELECT * FROM " + safe_name(row[0]))
    frame = pd.DataFrame(
        cursor.fetchall(), columns=[c[0] for c in cursor.description], dtype=object
    )
    return SourceContract.from_dict(spec["contract"]).validate(frame)


def load_source(db, source_id, version):
    with connection(db) as conn:
        row = conn.execute(
            "SELECT manifest_json FROM temporal_sources WHERE source_id=? AND version=?",
            (source_id, version),
        ).fetchone()
        if not row:
            raise KeyError((source_id, version))
        manifest = json.loads(row[0])
        frame = _read(conn, source_id, version)
        if frame_hash(frame) != manifest["data_sha256"]:
            raise ValueError("Source data integrity failure")
        return frame, manifest
