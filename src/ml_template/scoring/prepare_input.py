"""Snapshot and canonicalise raw, unlabelled input for an existing release."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from ml_template.data.canonicalize import canonicalize_frame
from ml_template.scoring.input_contract import validate_batch
from ml_template.scoring.release_artifacts import (
    read_json,
    sha256,
    verify,
    verify_runtime,
    write_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.output_dir is None:
        parser.error("--execute requires --output-dir (must not exist)")
    manifest = verify(args.release)
    verify_runtime(read_json(args.release / "runtime.json"))
    contract = read_json(args.release / "contract.json")
    spec = contract.get("input_spec")
    if spec is None:
        raise ValueError("release has no frozen input_spec; rebuild with integration patch")
    before = sha256(args.input)
    if args.input.suffix.lower() == ".csv":
        raw = pd.read_csv(args.input, dtype="string")
    elif args.input.suffix.lower() == ".parquet":
        raw = pd.read_parquet(args.input)
    else:
        raise ValueError("input must be CSV or Parquet")
    canonical_columns = set(raw.rename(columns=spec["canonical_names"]).columns)
    if canonical_columns & set(spec["forbidden_columns"]):
        raise ValueError("scoring input must not contain target/label columns")
    frame = canonicalize_frame(raw, spec, columns=spec["required_columns"])
    validate_batch(frame, contract, enforce_rates=True)
    if (
        "predictor.json" in manifest.get("files", {})
        and read_json(args.release / "predictor.json").get("task") == "regression"
    ):
        from ml_template.scoring.predictor import ReleasedPredictor

        ReleasedPredictor(args.release).prepare_features(frame)
    if sha256(args.input) != before:
        raise ValueError("input changed during preparation")
    report = {
        "release_id": manifest["release_id"],
        "input_sha256": before,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "null_counts": {c: int(n) for c, n in frame.isna().sum().items() if n},
        "registry_ingestion": False,
        "ready": True,
    }
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        snapshot = args.output_dir / ("raw_input" + args.input.suffix.lower())
        shutil.copyfile(args.input, snapshot)
        if sha256(snapshot) != before or sha256(args.input) != before:
            raise ValueError("input changed while snapshotting")
        output = args.output_dir / "data.parquet"
        frame.to_parquet(output, index=False)
        report["canonical_sha256"] = sha256(output)
        write_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
