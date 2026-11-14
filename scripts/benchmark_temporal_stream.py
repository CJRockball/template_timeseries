"Synthetic streaming-ingestion benchmark in a disposable directory. No real DB writes."

import argparse
import csv
import json
import platform
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from ml_template.temporal_stream.enrichment import assemble
from ml_template.temporal_stream.store import ingest_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=5000)
    args = p.parse_args()
    if args.rows < 1:
        p.error("--rows must be positive")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ts-benchmark-") as temp:
        temp = Path(temp)
        source = temp / "history.csv"
        dates = [(date(2011, 1, 1) + timedelta(days=i)).isoformat() for i in range(1941)]
        with source.open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["id", "series", "date", "sales"])
            for i in range(args.rows):
                w.writerow([str(i), str(i // 1941), dates[i % 1941], i % 11])
        contract = {
            "columns": {"id": "string", "series": "string", "date": "date", "sales": "int"},
            "key": ["id"],
            "temporal": {
                "series": ["series"],
                "time": "date",
                "frequency": "D",
                "gaps": "error",
                "target": "sales",
            },
        }
        begin = time.perf_counter()
        meta = ingest_csv(
            temp / "project.db", "benchmark", "v1", source, contract, batch_size=args.batch_size
        )
        ingestion_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        assembled = assemble(
            temp / "project.db",
            {"source_id": "benchmark", "version": "v1"},
            [],
            temp / "out",
            args.batch_size,
        )
        assembly_seconds = time.perf_counter() - begin
        result = {
            "rows": args.rows,
            "batch_size": args.batch_size,
            "ingest_seconds": ingestion_seconds,
            "assembly_seconds": assembly_seconds,
            "total_seconds": time.perf_counter() - started,
            "database_bytes": (temp / "project.db").stat().st_size,
            "max_insert_batch_rows": meta["max_insert_batch_rows"],
            "max_export_batch_rows": assembled["max_export_batch_rows"],
            "python": platform.python_version(),
        }
    status_path = Path("/proc/self/status")
    if status_path.exists():
        status = status_path.read_text()
        result["peak_rss_bytes"] = (
            int(
                next(line.split()[1] for line in status.splitlines() if line.startswith("VmHWM:"))
            )
            * 1024
        )
        result["rss_method"] = "Linux /proc/self/status VmHWM"
        print(json.dumps(result, indent=2))
        return
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = peak if platform.system() == "Darwin" else peak * 1024
    except ImportError:
        result["peak_rss_bytes"] = None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
