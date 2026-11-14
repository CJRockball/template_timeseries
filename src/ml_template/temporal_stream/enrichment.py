"Disk-backed SQL enrichment and bounded CSV export."

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .contracts import SourceContract, canonical_json, quote
from .store import FORMAT, connection, file_hash, verify_on_connection


def assemble(database, base, joins, output, batch_size=5000):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive batch size required")
    if set(base) != {"source_id", "version"}:
        raise ValueError("Base requires explicit source_id/version only")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".temporal-stream-", dir=output.parent))
    published = False
    try:
        with connection(database) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            table, meta = verify_on_connection(conn, base["source_id"], base["version"], batch_size)
            original = SourceContract.from_dict(meta["contract"])
            types = dict(original.columns)
            current = table
            lineage = []
            reports = []
            for number, recipe in enumerate(joins):
                allowed = {
                    "source_id",
                    "version",
                    "on",
                    "columns",
                    "mode",
                    "known_in_advance",
                    "available_at",
                    "origin",
                    "max_missing_fraction",
                }
                required = {"source_id", "version", "on", "columns", "mode"}
                if set(recipe) - allowed or not required <= set(recipe):
                    raise ValueError("Unknown/missing join setting")
                keys, selected = (recipe["on"], recipe["columns"])
                if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
                    raise ValueError("Unique nonempty join keys required")
                if not isinstance(selected, dict) or not selected:
                    raise ValueError("Source-to-output column mapping required")
                dests = list(selected.values())
                if len({x.casefold() for x in dests}) != len(dests) or any(
                    x.casefold() in {c.casefold() for c in types} for x in dests
                ):
                    raise ValueError("Enrichment column collision")
                for dest in dests:
                    quote(dest)
                    if dest.casefold().startswith("__ts_"):
                        raise ValueError("Reserved output prefix")
                right, rmeta = verify_on_connection(
                    conn, recipe["source_id"], recipe["version"], batch_size
                )
                rtypes = rmeta["contract"]["columns"]
                if (
                    not set([*keys, *selected]) <= rtypes.keys()
                    or not set(keys) <= original.columns.keys()
                ):
                    raise ValueError("Join keys must be base columns; selected columns must exist")
                if any(original.columns[k] != rtypes[k] for k in keys):
                    raise ValueError("Join key type mismatch")
                keylist = ",".join(quote(k) for k in keys)
                for source in [right, current]:
                    nulls = " OR ".join(quote(k) + " IS NULL" for k in keys)
                    if conn.execute(
                        "SELECT 1 FROM " + quote(source) + " WHERE " + nulls + " LIMIT 1"
                    ).fetchone():
                        raise ValueError("Null join key")
                mode = recipe["mode"]
                index_keys = list(keys)
                if mode == "exact":
                    if recipe.get("known_in_advance") is not True:
                        raise ValueError("Exact join requires known_in_advance assertion")
                    duplicates = keylist
                    join_sql = " AND ".join("r." + quote(k) + "=b." + quote(k) for k in keys)
                elif mode == "point_in_time":
                    available, origin = (recipe["available_at"], recipe["origin"])
                    if (
                        rtypes.get(available) != "utc_timestamp"
                        or original.columns.get(origin) != "utc_timestamp"
                    ):
                        raise ValueError("Availability and base origin must be UTC timestamps")
                    if available in keys:
                        raise ValueError("Availability must not be an equality key")
                    for source, column in [(right, available), (current, origin)]:
                        if conn.execute(
                            "SELECT 1 FROM "
                            + quote(source)
                            + " WHERE "
                            + quote(column)
                            + " IS NULL LIMIT 1"
                        ).fetchone():
                            raise ValueError("Null availability/origin")
                    index_keys.append(available)
                    duplicates = ",".join(quote(k) for k in index_keys)
                    matches = " AND ".join("s." + quote(k) + "=b." + quote(k) for k in keys)
                    join_sql = (
                        "r.rowid=(SELECT s.rowid FROM "
                        + quote(right)
                        + " s WHERE "
                        + matches
                        + " AND s."
                        + quote(available)
                        + "<=b."
                        + quote(origin)
                        + " ORDER BY s."
                        + quote(available)
                        + " DESC LIMIT 1)"
                    )
                else:
                    raise ValueError("Unknown join mode")
                idx = (
                    "tsi_"
                    + hashlib.sha256(canonical_json([right, index_keys]).encode()).hexdigest()
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS "
                    + quote(idx)
                    + " ON "
                    + quote(right)
                    + " ("
                    + ",".join(quote(k) for k in index_keys)
                    + ")"
                )
                if conn.execute(
                    "SELECT 1 FROM "
                    + quote(right)
                    + " GROUP BY "
                    + duplicates
                    + " HAVING COUNT(*)>1 LIMIT 1"
                ).fetchone():
                    raise ValueError("Ambiguous/many-to-many external join")
                next_table = "stage_" + str(number)
                expressions = ["b." + quote(c) for c in types]
                expressions += ["r." + quote(c) + " AS " + quote(d) for c, d in selected.items()]
                expressions.append("CASE WHEN r.rowid IS NULL THEN 0 ELSE 1 END AS __ts_match")
                conn.execute(
                    "CREATE TEMP TABLE "
                    + quote(next_table)
                    + " AS SELECT "
                    + ",".join(expressions)
                    + " FROM "
                    + quote(current)
                    + " b LEFT JOIN "
                    + quote(right)
                    + " r ON "
                    + join_sql
                )
                condition = " OR ".join(quote(c) + " IS NULL" for c in dests)
                count, unmatched, missing = conn.execute(
                    "SELECT COUNT(*), SUM(1-__ts_match), SUM(CASE WHEN "
                    + condition
                    + " THEN 1 ELSE 0 END) FROM "
                    + quote(next_table)
                ).fetchone()
                if count != meta["rows"]:
                    raise ValueError("Enrichment changed observation count")
                fraction = missing / count
                limit = float(recipe.get("max_missing_fraction", 0))
                if not 0 <= limit <= 1 or fraction > limit:
                    raise ValueError("Join missingness exceeds declared limit")
                reports.append(
                    {
                        "rows": count,
                        "unmatched_rows": unmatched,
                        "missing_feature_fraction": fraction,
                    }
                )
                lineage.append(
                    {
                        "recipe": recipe,
                        "source_fingerprint": rmeta["fingerprint"],
                        "source_data_sha256": rmeta["data_sha256"],
                    }
                )
                types.update({d: rtypes[c] for c, d in selected.items()})
                if current != table:
                    conn.execute("DROP TABLE " + quote(current))
                current = next_table
            columns = sorted(types)
            query = (
                "SELECT "
                + ",".join(quote(c) for c in columns)
                + " FROM "
                + quote(current)
                + " ORDER BY "
                + ",".join(quote(c) for c in original.order)
            )
            h = hashlib.sha256()
            h.update((canonical_json({"format": FORMAT, "columns": columns}) + "\n").encode())
            count, max_batch = (0, 0)
            with (staging / "data.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                cursor = conn.execute(query)
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    max_batch = max(max_batch, len(rows))
                    for row in rows:
                        h.update((canonical_json(list(row)) + "\n").encode())
                        writer.writerow(row)
                    count += len(rows)
            identity = {
                "format": FORMAT,
                "base": base,
                "base_fingerprint": meta["fingerprint"],
                "base_data_sha256": meta["data_sha256"],
                "joins": lineage,
                "data_sha256": h.hexdigest(),
            }
            assembly_id = (
                "assembly2_" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            )
            manifest = {
                **identity,
                "assembly_id": assembly_id,
                "rows": count,
                "columns": types,
                "reports": reports,
                "max_export_batch_rows": max_batch,
                "eligible_for_legacy_training": False,
                "csv_sha256": file_hash(staging / "data.csv"),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS temporal_assemblies_v2 (assembly_id TEXT PRIMARY KEY, identity_json TEXT NOT NULL)"
            )
            old = conn.execute(
                "SELECT identity_json FROM temporal_assemblies_v2 WHERE assembly_id=?",
                (assembly_id,),
            ).fetchone()
            if old and old[0] != canonical_json(identity):
                raise ValueError("Assembly identity conflict")
            conn.execute(
                "INSERT OR IGNORE INTO temporal_assemblies_v2 VALUES (?,?)",
                (assembly_id, canonical_json(identity)),
            )
            if output.exists():
                raise FileExistsError(output)
            staging.rename(output)
            published = True
        return manifest
    except BaseException:
        shutil.rmtree(output if published else staging, ignore_errors=True)
        raise
