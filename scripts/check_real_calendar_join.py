"""Check a real M5 calendar join using the installed temporal_stream bundle.

Run from the repo root. This writes test sources to the configured SQLite DB,
creates new export directories, and never changes the base sales snapshot.
Requires the streaming bundle and PyYAML. Not a point-in-time revision test.
"""

import argparse
import calendar
import csv
import hashlib
import io
import json
import uuid
from pathlib import Path

import yaml


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing prepared file differs: {path}; choose another --work-dir")
        return
    with path.open("xb") as handle:
        handle.write(content)


def main():
    from datetime import date

    from ml_template.temporal_stream.enrichment import assemble
    from ml_template.temporal_stream.store import ingest_csv, verify_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="The ingest YAML that successfully loaded smoke or small M5",
    )
    parser.add_argument("--calendar", type=Path, required=True, help="Original M5 calendar.csv")
    parser.add_argument("--work-dir", type=Path, default=Path("data/calendar_join_check"))
    args = parser.parse_args()
    cfg_path = args.base_config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    database = Path(cfg["database"])
    if not database.is_absolute():
        database = (cfg_path.parent / database).resolve()
    base = {"source_id": cfg["source_id"], "version": cfg["version"]}
    before = verify_source(database, base["source_id"], base["version"])
    if "weekday" in before["contract"]["columns"]:
        raise ValueError(
            "Base already contains weekday; use a different output alias in this helper"
        )
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(args.calendar)
    selected = io.StringIO(newline="")
    writer = csv.writer(selected)
    writer.writerow(["date", "weekday"])
    with args.calendar.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"date", "weekday"}.issubset(reader.fieldnames or []):
            raise ValueError("Original calendar must contain date and weekday")
        for row in reader:
            writer.writerow([row["date"], row["weekday"]])
    if file_hash(args.calendar) != source_hash:
        raise ValueError("Calendar changed during projection")
    prepared = work / "calendar_weekday.csv"
    write_immutable(prepared, selected.getvalue().encode("utf-8"))
    contract = {"columns": {"date": "date", "weekday": "string"}, "key": ["date"]}
    provenance = {"original_calendar_sha256": source_hash, "projection": ["date", "weekday"]}
    source_id = "m5_calendar_weekday_check"
    ingested = ingest_csv(database, source_id, "v1", prepared, contract, provenance)
    reused = ingest_csv(database, source_id, "v1", prepared, contract, provenance)
    if not reused["reused"]:
        raise ValueError("Unchanged calendar was not reused")
    join = {
        "source_id": source_id,
        "version": "v1",
        "on": ["date"],
        "columns": {"weekday": "weekday"},
        "mode": "exact",
        "known_in_advance": True,
        "max_missing_fraction": 0,
    }
    run = work / ("run_" + uuid.uuid4().hex[:12])
    result = assemble(database, base, [join], run)
    if result["rows"] != before["rows"] or result["reports"][0]["unmatched_rows"] != 0:
        raise ValueError("Row-count or coverage check failed")
    first_date = None
    count = 0
    with (run / "data.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected = calendar.day_name[date.fromisoformat(row["date"]).weekday()]
            if row["weekday"] != expected:
                raise ValueError(f"Incorrect weekday on {row['date']}")
            if first_date is None:
                first_date = row["date"]
            count += 1
    if count != before["rows"]:
        raise ValueError("Exported row count differs")

    # Negative test: remove a date actually used by the selected base profile.
    broken = run / "calendar_missing_date.csv"
    with (
        prepared.open(encoding="utf-8", newline="") as source,
        broken.open("x", encoding="utf-8", newline="") as dest,
    ):
        writer = csv.DictWriter(dest, fieldnames=["date", "weekday"])
        writer.writeheader()
        for row in csv.DictReader(source):
            if row["date"] != first_date:
                writer.writerow(row)
    bad_id = "m5_calendar_missing_" + file_hash(broken)[:16]
    ingest_csv(
        database,
        bad_id,
        "v1",
        broken,
        contract,
        {"negative_test": True, "removed_date": first_date},
    )
    failed_output = run / "must_not_publish"
    try:
        assemble(database, base, [{**join, "source_id": bad_id}], failed_output)
    except ValueError as exc:
        if "missingness" not in str(exc):
            raise
    else:
        raise ValueError("Negative test incorrectly succeeded")
    if failed_output.exists():
        raise ValueError("Failed assembly published output")
    after = verify_source(database, base["source_id"], base["version"])
    if after["data_sha256"] != before["data_sha256"]:
        raise ValueError("Base sales data changed")
    summary = {
        "status": "passed",
        "base": base,
        "rows": count,
        "unmatched_rows": 0,
        "weekday_checks": count,
        "unchanged_calendar_reused": reused["reused"],
        "missing_date_rejected": first_date,
        "base_data_unchanged": True,
        "calendar_rows": ingested["rows"],
        "export_directory": str(run),
    }
    (run / "check_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
