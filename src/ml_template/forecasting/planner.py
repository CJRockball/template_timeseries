"Disk-backed development-plan construction. Core has no Arrow/pandas dependency."

import hashlib
import json
import shutil
import sqlite3
import tempfile
from datetime import timedelta
from pathlib import Path

from .artifacts import FILES, file_hash, seal, verify_plan, write_line
from .policy import FORMAT, canonical, day, digest, effective_availability, validate_policy, windows
from .registry import register_plan


def _build_plan(records, silver, policy, output_root, database, batch_size=5000):
    """Internal seam for synthetic acceptance; public entry is io.build_from_silver."""
    policy = validate_policy(policy)
    silver = json.loads(canonical(silver))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive batch_size required")
    temporal = silver["temporal_schema"]
    if temporal["frequency"] != "D" or not temporal["series"]:
        raise ValueError("Daily panel semantics required")
    if type(temporal["target_delay_periods"]) is not int or temporal["target_delay_periods"] < 0:
        raise ValueError("Nonnegative target availability delay required")
    if not silver["silver_dataset_version_id"].startswith("silver_ts_"):
        raise ValueError("Temporal Silver ID required")
    if not Path(database).is_file():
        raise FileNotFoundError("Existing project SQLite database required")
    columns = [temporal["observation_id"], *temporal["series"], temporal["time"]]
    if temporal.get("available_at"):
        columns.append(temporal["available_at"])
    if len(columns) != len(set(columns)):
        raise ValueError("Distinct temporal identity roles required")
    if policy["series_selection"] and any(
        len(x) != len(temporal["series"]) for x in policy["series_selection"]
    ):
        raise ValueError("Composite series selection width mismatch")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".origin-build-", dir=output_root))
    published = None
    conn = None
    try:
        conn = sqlite3.connect(staging / "scratch.sqlite")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-32768")
        conn.execute(
            "CREATE TABLE observations (series TEXT NOT NULL, day TEXT NOT NULL, id TEXT NOT NULL UNIQUE, available TEXT NOT NULL, PRIMARY KEY(series,day)) WITHOUT ROWID"
        )
        batch = []
        count = 0
        maximum = 0
        with conn:
            for record in records:
                if set(record) != set(columns):
                    raise ValueError(
                        "Only declared identity/date/availability projection is allowed"
                    )
                values = [record[c] for c in temporal["series"]]
                identifier = record[temporal["observation_id"]]
                if identifier is None or any(v is None for v in values):
                    raise ValueError("Null temporal key")
                date = day(record[temporal["time"]]).isoformat()
                actual = record.get(temporal.get("available_at"))
                if temporal.get("available_at") and actual is None:
                    raise ValueError("Missing declared availability")
                batch.append(
                    (
                        canonical(values),
                        date,
                        canonical(identifier),
                        effective_availability(date, actual, policy, temporal),
                    )
                )
                count += 1
                if len(batch) == batch_size:
                    conn.executemany("INSERT INTO observations VALUES (?,?,?,?)", batch)
                    maximum = max(maximum, len(batch))
                    batch.clear()
            if batch:
                conn.executemany("INSERT INTO observations VALUES (?,?,?,?)", batch)
                maximum = max(maximum, len(batch))
        if count == 0 or count != silver["row_count"]:
            raise ValueError("Silver projected row count mismatch")
        first, last = conn.execute("SELECT MIN(day),MAX(day) FROM observations").fetchone()
        holdout_start = (day(last) - timedelta(days=policy["holdout_periods"] - 1)).isoformat()
        if holdout_start <= first:
            raise ValueError("Holdout leaves no development history")
        all_windows = [
            windows(origin, policy, temporal) for origin in policy["development_origins"]
        ]
        if any(w["target_date"] >= holdout_start for w in all_windows):
            raise ValueError("Development request overlaps reserved holdout")
        if any(w["refit_end"] >= holdout_start for w in all_windows):
            raise ValueError("Development history overlaps reserved holdout")
        if policy["series_selection"] is not None:
            conn.execute("CREATE TEMP TABLE selected (series TEXT PRIMARY KEY)")
            for values in policy["series_selection"]:
                key = canonical(values)
                if (
                    conn.execute(
                        "SELECT 1 FROM observations WHERE series=? LIMIT 1", (key,)
                    ).fetchone()
                    is None
                ):
                    raise ValueError("Selected series not found: " + key)
                conn.execute("INSERT INTO selected VALUES (?)", (key,))
            series_query = "SELECT series FROM selected ORDER BY series"
        else:
            series_query = "SELECT DISTINCT series FROM observations ORDER BY series"
        index = hashlib.sha256()
        for row in conn.execute(
            "SELECT series,day,id,available FROM observations ORDER BY series,day"
        ):
            index.update((canonical(list(row)) + "\n").encode())
        identity = {
            "format": FORMAT,
            "silver": silver,
            "policy": policy,
            "build_batch_size": batch_size,
            "windows": all_windows,
            "index_sha256": index.hexdigest(),
            "holdout": {
                "start": holdout_start,
                "end": last,
                "periods": policy["holdout_periods"],
                "reserved": True,
                "requests_materialized": False,
            },
            "training_sample_origin_rule": "previous_business_day_end"
            if temporal["origin_policy"] == "end_of_business_day"
            else "target_business_day_start",
            "availability_rule": "max(declared_utc_availability,end_of_business_day(date+delay))",
            "executor_implemented": False,
        }
        pid = "origin_" + digest(identity)
        request_count = 0
        exclusions = {}
        pairs = 0
        with (
            (staging / "origins.jsonl").open("w", encoding="utf-8") as origins_file,
            (staging / "eligibility.jsonl").open("w", encoding="utf-8") as eligibility_file,
            (staging / "requests.jsonl").open("w", encoding="utf-8") as requests_file,
        ):
            for window in all_windows:
                write_line(origins_file, window)
                eligible_at_origin = 0
                for (key,) in conn.execute(series_query):
                    fit = conn.execute(
                        "SELECT COUNT(*) FROM observations WHERE series=? AND day BETWEEN ? AND ? AND available<=?",
                        (key, window["fit_start"], window["fit_end"], window["fit_available_by"]),
                    ).fetchone()[0]
                    stop = conn.execute(
                        "SELECT COUNT(*) FROM observations WHERE series=? AND day BETWEEN ? AND ? AND available<=?",
                        (
                            key,
                            window["stopping_start"],
                            window["stopping_end"],
                            window["issued_at"],
                        ),
                    ).fetchone()[0]
                    refit = conn.execute(
                        "SELECT COUNT(*) FROM observations WHERE series=? AND day BETWEEN ? AND ? AND available<=?",
                        (key, window["refit_start"], window["refit_end"], window["issued_at"]),
                    ).fetchone()[0]
                    target = conn.execute(
                        "SELECT id FROM observations WHERE series=? AND day=?",
                        (key, window["target_date"]),
                    ).fetchone()
                    reasons = []
                    if fit < policy["min_train_observations"]:
                        reasons.append("insufficient_fit_history")
                    if stop < policy["stopping_periods"]:
                        reasons.append("incomplete_stopping_window")
                    if target is None:
                        reasons.append("missing_forecast_observation")
                    pairs += 1
                    for reason in reasons:
                        exclusions[reason] = exclusions.get(reason, 0) + 1
                    values = json.loads(key)
                    write_line(
                        eligibility_file,
                        {
                            "origin_date": window["origin_date"],
                            "series": values,
                            "fit_count": fit,
                            "stopping_count": stop,
                            "refit_count": refit,
                            "eligible": not reasons,
                            "reasons": reasons,
                        },
                    )
                    if reasons:
                        continue
                    request = {
                        "plan_id": pid,
                        "purpose": "development",
                        "series": values,
                        "origin_date": window["origin_date"],
                        "issued_at": window["issued_at"],
                        "target_date": window["target_date"],
                        "horizon": 1,
                        "observation_id": json.loads(target[0]),
                    }
                    request["request_id"] = "request_" + digest(request)
                    write_line(requests_file, request)
                    request_count += 1
                    eligible_at_origin += 1
                if eligible_at_origin == 0:
                    raise ValueError("No eligible series for origin " + window["origin_date"])
        conn.close()
        conn = None
        (staging / "scratch.sqlite").unlink()
        report = {
            "passed": True,
            "requests": request_count,
            "origin_count": len(all_windows),
            "series_origin_pairs": pairs,
            "exclusions": exclusions,
            "source_observations": count,
            "max_index_batch_rows": maximum,
            "target_values_read": False,
            "holdout_requests_materialized": False,
        }
        (staging / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        inventory = {
            name: {"bytes": (staging / name).stat().st_size, "sha256": file_hash(staging / name)}
            for name in FILES
        }
        manifest = seal(
            {
                "format": FORMAT,
                "plan_id": pid,
                "identity": identity,
                "files": inventory,
                "validation": report,
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_plan(staging)
        final = output_root / pid
        if final.exists():
            previous = verify_plan(final)
            if previous["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("Immutable plan conflict; use original batch size for exact reuse")
            register_plan(database, final)
            shutil.rmtree(staging)
            return {**previous, "plan_path": str(final), "reused": True}
        staging.rename(final)
        published = final
        register_plan(database, final)
        return {**manifest, "plan_path": str(final), "reused": False}
    except BaseException:
        if conn is not None:
            conn.close()
        shutil.rmtree(published if published else staging, ignore_errors=True)
        raise
