"""Optional M5-only structuring; no database writes or project imports."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history",
        type=Path,
        required=True,
        help="Notebook history_long.csv, NOT concatenated with train/holdout",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    df = pd.read_csv(a.history, dtype={"series_id": str, "store_id": str, "item_id": str})
    cols = ["series_id", "date", "sales", "store_id", "item_id", "wm_yr_wk"]
    df = df[cols].sort_values(["series_id", "date"]).reset_index(drop=True)
    if df.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate observations")
    df.insert(
        0,
        "observation_id",
        [
            hashlib.sha256(json.dumps([s, d], ensure_ascii=False).encode()).hexdigest()
            for s, d in zip(df.series_id, df.date, strict=True)
        ],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(df),
                "series": int(df.series_id.nunique()),
                "input_sha256": hashlib.sha256(a.history.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
