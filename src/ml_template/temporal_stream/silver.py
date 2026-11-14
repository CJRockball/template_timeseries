"Publish streaming inputs as versioned multi-file temporal Silver Parquet."

import argparse
import csv
import json
import shutil
import sqlite3
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from .silver_schema import FORMAT, TemporalSchema, canonical, file_hash, positive, quote, sha


def source_rows(database, reference, batch_size):
    from .contracts import SourceContract
    from .store import connection, source_query, verify_on_connection

    with connection(database) as conn:
        conn.execute("BEGIN")
        table, meta = verify_on_connection(
            conn, reference["source_id"], reference["version"], batch_size
        )
        spec = SourceContract.from_dict(meta["contract"])
        cursor = conn.execute(source_query(table, spec))
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                yield dict(zip(spec.columns, row, strict=True))


def input_identity(database, input_spec, schema):
    from .store import connection, verify_source

    if set(input_spec) == {"source"}:
        ref = input_spec["source"]
        if set(ref) != {"source_id", "version"}:
            raise ValueError("Explicit source_id/version required")
        meta = verify_source(database, ref["source_id"], ref["version"])
        if meta["contract"]["columns"] != schema.spec["columns"]:
            raise ValueError("Source types differ from temporal schema")
        return {
            "kind": "source",
            **ref,
            "fingerprint": meta["fingerprint"],
            "data_sha256": meta["data_sha256"],
            "rows": meta["rows"],
            "hash_format": meta["hash_format"],
        }
    if set(input_spec) == {"assembly"}:
        folder = Path(input_spec["assembly"]).resolve()
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        identity = {
            key: meta[key]
            for key in [
                "format",
                "base",
                "base_fingerprint",
                "base_data_sha256",
                "joins",
                "data_sha256",
            ]
        }
        if meta["assembly_id"] != "assembly2_" + sha(identity):
            raise ValueError("Assembly identity mismatch")
        if (
            meta["columns"] != schema.spec["columns"]
            or file_hash(folder / "data.csv") != meta["csv_sha256"]
        ):
            raise ValueError("Assembly schema/CSV integrity failure")
        with connection(database) as conn:
            row = conn.execute(
                "SELECT identity_json FROM temporal_assemblies_v2 WHERE assembly_id=?",
                (meta["assembly_id"],),
            ).fetchone()
            if not row or json.loads(row[0]) != identity:
                raise ValueError("Assembly is not registered in this database")
        return {
            "kind": "assembly",
            "assembly_id": meta["assembly_id"],
            "data_sha256": meta["data_sha256"],
            "csv_sha256": meta["csv_sha256"],
            "manifest_sha256": file_hash(folder / "manifest.json"),
            "rows": meta["rows"],
            "hash_format": meta["format"],
        }
    raise ValueError("Input must reference exactly one source or assembly")


def assembly_rows(folder, schema):
    with (Path(folder) / "data.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != schema.columns:
            raise ValueError("Assembly column header mismatch")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed assembly CSV")
            yield row


def stage_rows(path, records, schema, batch_size, input_hash_format=None):
    import hashlib

    conn = sqlite3.connect(path)
    original_hasher = hashlib.sha256()
    original_hasher.update(
        (canonical({"format": input_hash_format, "columns": schema.columns}) + "\n").encode()
    )
    try:
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-32768")
        types = {"int": "INTEGER", "float": "REAL"}
        defs = [
            quote(c)
            + " "
            + types.get(schema.spec["columns"][c], "TEXT")
            + ("" if c in schema.spec["nullable"] else " NOT NULL")
            for c in schema.columns
        ]
        defs += [
            "UNIQUE (" + quote(schema.spec["observation_id"]) + ")",
            "UNIQUE (" + ",".join(quote(c) for c in schema.order) + ")",
        ]
        conn.execute("CREATE TABLE observations (" + ",".join(defs) + ")")
        insert = "INSERT INTO observations VALUES (" + ",".join("?" for _ in schema.columns) + ")"
        batch, count, maximum = ([], 0, 0)
        with conn:
            for record in records:
                parsed = schema.parse(record)
                original_hasher.update((canonical(list(parsed)) + "\n").encode())
                batch.append(parsed)
                count += 1
                if len(batch) == batch_size:
                    conn.executemany(insert, batch)
                    maximum = max(maximum, len(batch))
                    batch.clear()
            if batch:
                conn.executemany(insert, batch)
                maximum = max(maximum, len(batch))
        if count == 0:
            raise ValueError("Empty temporal dataset")
        if schema.spec["gaps"] == "error":
            query = (
                "SELECT "
                + ",".join(quote(c) for c in schema.order)
                + " FROM observations ORDER BY "
                + ",".join(quote(c) for c in schema.order)
            )
            previous = None
            for row in conn.execute(query):
                key, day = (tuple(row[:-1]), date.fromisoformat(row[-1]))
                if previous is not None and key == previous[0] and ((day - previous[1]).days != 1):
                    raise ValueError("Undeclared daily gap")
                previous = (key, day)
        return (count, maximum, original_hasher.hexdigest())
    finally:
        conn.close()


def build(
    database,
    input_spec,
    schema_spec,
    output_root,
    *,
    batch_size=5000,
    max_rows_per_file=250000,
    compression="zstd",
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ml_template.db.temporal_silver_registry import register_temporal_silver

    from .silver_io import validate_parquet
    from .silver_manifest import seal, verify_inventory

    positive(batch_size, "batch_size")
    positive(max_rows_per_file, "max_rows_per_file")
    if compression not in {"zstd", "snappy", "none"}:
        raise ValueError("Unsupported compression")
    schema = TemporalSchema.from_dict(schema_spec)
    database = Path(database).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    original = input_identity(database, input_spec, schema)
    staging = Path(tempfile.mkdtemp(prefix=".silver-build-", dir=output_root))
    published = None
    try:
        records = (
            source_rows(database, input_spec["source"], batch_size)
            if original["kind"] == "source"
            else assembly_rows(input_spec["assembly"], schema)
        )
        count, maximum, input_logical_hash = stage_rows(
            staging / "scratch.sqlite", records, schema, batch_size, original["hash_format"]
        )
        if input_logical_hash != original["data_sha256"]:
            raise ValueError("Input logical data hash mismatch")
        if count != original["rows"]:
            raise ValueError("Input row count mismatch")
        if input_identity(database, input_spec, schema) != original:
            raise ValueError("Input changed during Silver build")
        data = staging / "data"
        data.mkdir()
        conn = sqlite3.connect(staging / "scratch.sqlite")
        inventory = []
        writer = None
        h = schema.new_hasher()
        try:
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA cache_size=-32768")
            cursor = conn.execute(
                "SELECT "
                + ",".join(quote(c) for c in schema.columns)
                + " FROM observations ORDER BY "
                + ",".join(quote(c) for c in schema.order)
            )
            part_rows, part = (0, None)
            while True:
                capacity = (
                    min(batch_size, max_rows_per_file - part_rows)
                    if writer
                    else min(batch_size, max_rows_per_file)
                )
                rows = cursor.fetchmany(capacity)
                if not rows:
                    break
                if writer is None:
                    part = data / f"part-{len(inventory):06d}.parquet"
                    writer = pq.ParquetWriter(
                        part,
                        schema.arrow_schema(),
                        compression=None if compression == "none" else compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                    part_rows = 0
                for row in rows:
                    h.update((canonical(list(row)) + "\n").encode())
                writer.write_table(schema.arrow_table(rows), row_group_size=batch_size)
                part_rows += len(rows)
                if part_rows == max_rows_per_file:
                    writer.close()
                    writer = None
                    inventory.append(
                        {
                            "path": part.relative_to(staging).as_posix(),
                            "rows": part_rows,
                            "bytes": part.stat().st_size,
                            "sha256": file_hash(part),
                        }
                    )
            if writer is not None:
                writer.close()
                writer = None
                inventory.append(
                    {
                        "path": part.relative_to(staging).as_posix(),
                        "rows": part_rows,
                        "bytes": part.stat().st_size,
                        "sha256": file_hash(part),
                    }
                )
        finally:
            if writer is not None:
                writer.close()
            conn.close()
        (staging / "scratch.sqlite").unlink()
        identity = {
            "format": FORMAT,
            "input": original,
            "temporal_schema": schema.spec,
            "logical_data_sha256": h.hexdigest(),
            "layout": {
                "batch_size": batch_size,
                "max_rows_per_file": max_rows_per_file,
                "compression": compression,
                "pyarrow_version": pa.__version__,
            },
        }
        sid = "silver_ts_" + sha(identity)
        manifest = {
            "format": FORMAT,
            "dataset_kind": "temporal",
            "silver_dataset_version_id": sid,
            "identity": identity,
            "input": original,
            "temporal_schema": schema.spec,
            "roles": schema.roles,
            "columns": schema.spec["columns"],
            "order_by": schema.order,
            "row_count": count,
            "logical_data_sha256": h.hexdigest(),
            "files": inventory,
            "validation_passed": True,
            "eligible_for_legacy_training": False,
            "created_at": datetime.now(UTC).isoformat(),
        }
        validate_parquet(staging, manifest, batch_size)
        validation = {
            "passed": True,
            "rows": count,
            "max_stage_batch_rows": maximum,
            "checks": [
                "strict_types",
                "unique_observation",
                "unique_series_date",
                "declared_gaps",
                "parquet_logical_round_trip",
            ],
        }
        (staging / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
        manifest["validation_sha256"] = file_hash(staging / "validation.json")
        manifest = seal(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        final = output_root / sid
        if final.exists():
            previous = verify_inventory(final)
            if previous["identity"] != identity:
                raise ValueError("Immutable Silver identity conflict")
            register_temporal_silver(database, final / "manifest.json")
            shutil.rmtree(staging)
            return {**previous, "reused": True, "dataset_path": str(final)}
        staging.rename(final)
        published = final
        register_temporal_silver(database, final / "manifest.json")
        return {**manifest, "reused": False, "dataset_path": str(final)}
    except BaseException:
        shutil.rmtree(published if published else staging, ignore_errors=True)
        raise


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--ingest-config", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--target-delay-periods", type=int, required=True)
    init.add_argument(
        "--origin-policy", choices=["start_of_business_day", "end_of_business_day"], required=True
    )
    for name in ["build", "verify"]:
        child = sub.add_parser(name)
        child.add_argument(
            "--config", type=Path, required=True
        ) if name == "build" else child.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        source = args.ingest_config.resolve()
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        c = original["contract"]
        t = c["temporal"]
        if len(c["key"]) != 1:
            raise ValueError("Initial bridge requires one observation ID; prepare it externally")
        db = Path(original["database"])
        if not db.is_absolute():
            db = (source.parent / db).resolve()
        spec = {
            "columns": c["columns"],
            "nullable": c.get("nullable", []),
            "observation_id": c["key"][0],
            "series": t["series"],
            "time": t["time"],
            "target": t["target"],
            "features": [],
            "known_future": [],
            "frequency": t["frequency"],
            "gaps": t.get("gaps", "error"),
            "target_delay_periods": args.target_delay_periods,
            "origin_policy": args.origin_policy,
        }
        TemporalSchema.from_dict(spec)
        cfg = {
            "database": str(db),
            "input": {
                "source": {"source_id": original["source_id"], "version": original["version"]}
            },
            "schema": spec,
            "output_root": str((db.parent / "silver_temporal").resolve()),
            "batch_size": 5000,
            "max_rows_per_file": 250000,
            "compression": "zstd",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print(str(args.output.resolve()))
        return
    if args.command == "verify":
        from .silver_io import validate_parquet
        from .silver_manifest import verify_inventory

        manifest = verify_inventory(args.dataset)
        validate_parquet(args.dataset, manifest)
        print(
            json.dumps(
                {
                    "passed": True,
                    "rows": manifest["row_count"],
                    "silver_dataset_version_id": manifest["silver_dataset_version_id"],
                },
                indent=2,
            )
        )
        return
    path = args.config.resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"database", "input", "schema", "output_root"}
    if set(cfg) - (
        required | {"batch_size", "max_rows_per_file", "compression"}
    ) or not required <= set(cfg):
        raise ValueError("Unknown/missing Silver configuration field")

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent / p

    input_spec = dict(cfg["input"])
    if "assembly" in input_spec:
        input_spec["assembly"] = str(resolve(input_spec["assembly"]))
    result = build(
        resolve(cfg["database"]),
        input_spec,
        cfg["schema"],
        resolve(cfg["output_root"]),
        batch_size=cfg.get("batch_size", 5000),
        max_rows_per_file=cfg.get("max_rows_per_file", 250000),
        compression=cfg.get("compression", "zstd"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
