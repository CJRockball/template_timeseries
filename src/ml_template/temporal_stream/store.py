"Bounded-Python-memory ingestion and verification into isolated SQLite v2 tables."

import csv
import hashlib
import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

from .contracts import SourceContract, canonical_json, quote

FORMAT = "typed-jsonl-sha256-v2"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DigestReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


@contextmanager
def connection(database):
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-32768")
    try:
        yield conn
    finally:
        conn.close()


def setup(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS temporal_sources_v2 (\n        source_id TEXT NOT NULL, version TEXT NOT NULL, table_name TEXT NOT NULL UNIQUE,\n        manifest_json TEXT NOT NULL, PRIMARY KEY (source_id,version))"
    )


def metadata(conn, source_id, version):
    row = conn.execute(
        "SELECT table_name,manifest_json FROM temporal_sources_v2 WHERE source_id=? AND version=?",
        (source_id, version),
    ).fetchone()
    if row is None:
        raise KeyError((source_id, version))
    return (row[0], json.loads(row[1]))


def digest_query(conn, query, columns, batch_size=5000):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive batch size required")
    h = hashlib.sha256()
    h.update((canonical_json({"format": FORMAT, "columns": columns}) + "\n").encode())
    cursor = conn.execute(query)
    count = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            h.update((canonical_json(list(row)) + "\n").encode())
            count += 1
    return (h.hexdigest(), count)


def source_query(table, contract):
    columns = ",".join(quote(c) for c in contract.columns)
    order = ",".join(quote(c) for c in contract.order)
    return f"SELECT {columns} FROM {quote(table)} ORDER BY {order}"


def verify_on_connection(conn, source_id, version, batch_size=5000):
    table, meta = metadata(conn, source_id, version)
    contract = SourceContract.from_dict(meta["contract"])
    actual, count = digest_query(
        conn, source_query(table, contract), list(contract.columns), batch_size
    )
    if actual != meta["data_sha256"] or count != meta["rows"]:
        raise ValueError("Source data integrity failure")
    return (table, meta)


def ingest_csv(database, source_id, version, path, contract, provenance=None, batch_size=5000):
    if (
        not isinstance(source_id, str)
        or not source_id
        or (not isinstance(version, str))
        or (not version)
    ):
        raise ValueError("Explicit nonempty string source_id/version required")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    contract = SourceContract.from_dict(contract) if isinstance(contract, dict) else contract
    path = Path(path)
    original_hash = file_hash(path)
    identity = {
        "source_sha256": original_hash,
        "contract": contract.to_dict(),
        "provenance": provenance or {},
        "hash_format": FORMAT,
    }
    fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    table = "ts2_" + hashlib.sha256(canonical_json([source_id, version]).encode()).hexdigest()
    with connection(database) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        setup(conn)
        old = conn.execute(
            "SELECT 1 FROM temporal_sources_v2 WHERE source_id=? AND version=?",
            (source_id, version),
        ).fetchone()
        if old:
            _, prior = metadata(conn, source_id, version)
            if prior["fingerprint"] != fingerprint:
                raise ValueError("Immutable version conflict; use a new version")
            verify_on_connection(conn, source_id, version, batch_size)
            return {**prior, "reused": True}
        types = {"int": "INTEGER", "float": "REAL"}
        definitions = [
            quote(c) + " " + types.get(k, "TEXT") + ("" if c in contract.nullable else " NOT NULL")
            for c, k in contract.columns.items()
        ]
        definitions.append("UNIQUE (" + ",".join(quote(c) for c in contract.key) + ")")
        if contract.temporal:
            keys = [*contract.temporal["series"], contract.temporal["time"]]
            definitions.append("UNIQUE (" + ",".join(quote(c) for c in keys) + ")")
        conn.execute("CREATE TABLE " + quote(table) + " (" + ",".join(definitions) + ")")
        statement = (
            "INSERT INTO "
            + quote(table)
            + " VALUES ("
            + ",".join("?" for _ in contract.columns)
            + ")"
        )
        count = 0
        max_batch = 0
        batch = []
        with path.open("rb") as raw:
            reader = DigestReader(raw)
            with io.TextIOWrapper(
                io.BufferedReader(reader), encoding="utf-8-sig", newline=""
            ) as handle:
                csv_reader = csv.DictReader(handle)
                header = csv_reader.fieldnames
                if (
                    not header
                    or len(header) != len(set(header))
                    or set(header) != set(contract.columns)
                ):
                    raise ValueError("Input header must exactly match source contract")
                for record in csv_reader:
                    if None in record or any(v is None for v in record.values()):
                        raise ValueError("Malformed CSV row width")
                    try:
                        batch.append(contract.parse(record))
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"CSV line {csv_reader.line_num}: {exc}") from exc
                    count += 1
                    if len(batch) == batch_size:
                        conn.executemany(statement, batch)
                        max_batch = max(max_batch, len(batch))
                        batch.clear()
                if batch:
                    conn.executemany(statement, batch)
                    max_batch = max(max_batch, len(batch))
                    batch.clear()
            parsed_hash = reader.digest.hexdigest()
        if count == 0:
            raise ValueError("Empty source")
        if parsed_hash != original_hash:
            raise ValueError("Source changed during ingestion")
        if contract.temporal and contract.temporal.get("gaps", "error") == "error":
            t = contract.temporal
            keys = [*t["series"], t["time"]]
            cursor = conn.execute(
                "SELECT "
                + ",".join(quote(c) for c in keys)
                + " FROM "
                + quote(table)
                + " ORDER BY "
                + ",".join(quote(c) for c in keys)
            )
            previous_series = None
            previous_date = None
            for record in cursor:
                series, current = (tuple(record[:-1]), date.fromisoformat(record[-1]))
                if series == previous_series and (current - previous_date).days != 1:
                    raise ValueError("Undeclared missing daily period")
                previous_series, previous_date = (series, current)
        data_hash, actual = digest_query(
            conn, source_query(table, contract), list(contract.columns), batch_size
        )
        if count != actual:
            raise ValueError("Inserted row count mismatch")
        meta = {
            **identity,
            "source_id": source_id,
            "version": version,
            "fingerprint": fingerprint,
            "rows": count,
            "data_sha256": data_hash,
            "max_insert_batch_rows": max_batch,
            "created_at": datetime.now(UTC).isoformat(),
        }
        conn.execute(
            "INSERT INTO temporal_sources_v2 VALUES (?,?,?,?)",
            (source_id, version, table, canonical_json(meta)),
        )
    return {**meta, "reused": False}


def verify_source(database, source_id, version, batch_size=5000):
    with connection(database) as conn:
        conn.execute("BEGIN")
        return verify_on_connection(conn, source_id, version, batch_size)[1]


def iter_source(database, source_id, version, batch_size=5000):
    if batch_size < 1:
        raise ValueError("Positive batch_size required")
    with connection(database) as conn:
        conn.execute("BEGIN")
        table, meta = verify_on_connection(conn, source_id, version, batch_size)
        contract = SourceContract.from_dict(meta["contract"])
        cursor = conn.execute(source_query(table, contract))
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            yield batch
