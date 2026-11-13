"""Create a feature-only development sample and CSV reference skeleton; no labels."""

import argparse
from pathlib import Path

import pandas as pd

from ml_template.paths import DATA_SILVER, DATA_SPLITS
from ml_template.scoring.release_artifacts import read_json, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-column", default="strength")
    p.add_argument("--rows", type=int, default=10)
    args = p.parse_args()
    if (
        args.rows < 1
        or args.output.exists()
        or args.reference.exists()
        or args.output.resolve() == args.reference.resolve()
    ):
        raise ValueError("Use positive rows and two distinct, new output paths")
    verify(args.release)
    contract = read_json(args.release / "contract.json")
    cfg = read_json(args.release / "config.json")
    evidence = read_json(args.release / "evidence.json")
    key = contract["key_col"]
    if key == args.output_column:
        raise ValueError("Key and output column must differ")
    columns = [key, *[item["name"] for item in contract["source_schema"]]]
    assignment = pd.read_parquet(
        DATA_SPLITS / evidence["evaluation_spec"]["expected_split_id"] / "assignments.parquet"
    )
    keys = assignment.loc[assignment.partition.eq("dev"), key].head(args.rows).tolist()
    if not keys:
        raise ValueError("No development rows")
    data = cfg["data"]
    path = DATA_SILVER / data["dataset_name"] / data["silver_dataset_version_id"] / "data.parquet"
    frame = pd.read_parquet(path, columns=columns, filters=[(key, "in", keys)])
    if set(frame[key]) != set(keys):
        raise ValueError("Demo key coverage differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    with args.reference.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame({key: frame[key], args.output_column: 0.0}).to_csv(stream, index=False)
    print(
        f"Created {len(frame)} feature-only DEVELOPMENT rows; this is a wiring demo, not an accuracy test"
    )


if __name__ == "__main__":
    main()
