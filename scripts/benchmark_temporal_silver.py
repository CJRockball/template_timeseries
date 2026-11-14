"Disposable native-Arrow Silver benchmark. No real project database is touched."

import argparse
import csv
import json
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from ml_template.temporal_stream.silver import build
from ml_template.temporal_stream.silver_io import load_temporal_silver
from ml_template.temporal_stream.store import ingest_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--rows-per-file", type=int, default=250000)
    args = parser.parse_args()
    if min(args.rows, args.batch_size, args.rows_per_file) < 1:
        parser.error("All counts must be positive")
    with tempfile.TemporaryDirectory(prefix="silver-benchmark-") as temp:
        temp = Path(temp)
        dates = [(date(2011, 1, 1) + timedelta(days=i)).isoformat() for i in range(1941)]
        raw = temp / "raw.csv"
        with raw.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "series", "date", "y"])
            for i in range(args.rows):
                writer.writerow([str(i), "s" + str(i // 1941), dates[i % 1941], i % 11])
        columns = {"id": "string", "series": "string", "date": "date", "y": "int"}
        contract = {
            "columns": columns,
            "key": ["id"],
            "temporal": {
                "series": ["series"],
                "time": "date",
                "target": "y",
                "frequency": "D",
                "gaps": "error",
            },
        }
        database = temp / "project.db"
        ingest_csv(database, "synthetic", "v1", raw, contract, batch_size=args.batch_size)
        schema = {
            "columns": columns,
            "observation_id": "id",
            "series": ["series"],
            "time": "date",
            "target": "y",
            "features": [],
            "known_future": [],
            "frequency": "D",
            "gaps": "error",
            "target_delay_periods": 0,
            "origin_policy": "end_of_business_day",
        }
        started = time.perf_counter()
        result = build(
            database,
            {"source": {"source_id": "synthetic", "version": "v1"}},
            schema,
            temp / "silver",
            batch_size=args.batch_size,
            max_rows_per_file=args.rows_per_file,
        )
        publish_seconds = time.perf_counter() - started
        started = time.perf_counter()
        maximum, count = (0, 0)
        for batch in load_temporal_silver(result["dataset_path"]).scan_batches(
            batch_size=args.batch_size
        ):
            maximum = max(maximum, batch.num_rows)
            count += batch.num_rows
        if count != args.rows or maximum > args.batch_size:
            raise ValueError("Native scan acceptance failed")
        metrics = {
            "rows": count,
            "publish_seconds": publish_seconds,
            "verified_scan_seconds": time.perf_counter() - started,
            "max_read_batch_rows": maximum,
            "parquet_files": len(result["files"]),
            "parquet_bytes": sum(f["bytes"] for f in result["files"]),
            "pyarrow_version": result["identity"]["layout"]["pyarrow_version"],
        }
        status = Path("/proc/self/status")
        if status.exists():
            value = next(

                    line.split()[1]
                    for line in status.read_text().splitlines()
                    if line.startswith("VmHWM:")

            )
            metrics["peak_rss_bytes"] = int(value) * 1024
            metrics["rss_method"] = "Linux /proc/self/status VmHWM, includes source ingestion"
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
