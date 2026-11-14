"M5-only streaming projection from long notebook history OR original wide sales.\n\nNo database writes; output is a generic long CSV. Static M5 price/calendar joins\nare deliberately excluded. For --wide, calendar.csv must supply d,date,wm_yr_wk.\n"

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wide", action="store_true")
    p.add_argument("--calendar", type=Path)
    args = p.parse_args(argv)
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".manifest.json").exists()
    ):
        raise FileExistsError(args.output)
    if args.wide and args.calendar is None:
        p.error("--wide requires --calendar")
    source_hash_before = sha(args.input)
    calendar_hash_before = sha(args.calendar) if args.wide else None
    calendar = {}
    if args.wide:
        with args.calendar.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["d"] in calendar:
                    raise ValueError("Duplicate calendar day")
                calendar[row["d"]] = (row["date"], row["wm_yr_wk"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with (
            args.input.open(encoding="utf-8-sig", newline="") as source,
            args.output.open("x", encoding="utf-8", newline="") as dest,
        ):
            reader = csv.DictReader(source)
            writer = csv.writer(dest)
            writer.writerow(
                ["observation_id", "series_id", "date", "sales", "store_id", "item_id", "wm_yr_wk"]
            )
            days = sorted(
                [c for c in reader.fieldnames if re.fullmatch("d_\\d+", c)],
                key=lambda c: int(c[2:]),
            )
            if args.wide and (not days):
                raise ValueError("No daily sales columns")
            for row in reader:
                if args.wide:
                    series = json.dumps([row["item_id"], row["store_id"]], separators=(",", ":"))
                    observations = [(calendar[d][0], row[d], calendar[d][1]) for d in days]
                else:
                    series = row["series_id"]
                    observations = [(row["date"], row["sales"], row["wm_yr_wk"])]
                for day, sales, week in observations:
                    oid = hashlib.sha256(
                        json.dumps([series, day], separators=(",", ":")).encode()
                    ).hexdigest()
                    writer.writerow(
                        [oid, series, day, sales, row["store_id"], row["item_id"], week]
                    )
                    count += 1
        if sha(args.input) != source_hash_before or (
            args.wide and sha(args.calendar) != calendar_hash_before
        ):
            raise ValueError("Source changed during preparation")
        meta = {
            "rows": count,
            "source_sha256": sha(args.input),
            "output_sha256": sha(args.output),
            "calendar_sha256": sha(args.calendar) if args.wide else None,
            "mode": "wide" if args.wide else "long",
            "script": "prepare_m5_stream.py v2",
        }
        args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
