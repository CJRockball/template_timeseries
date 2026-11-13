"""Export keyed probabilities to a generic two-column CSV; never fit or predict."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.scoring.release_artifacts import sha256, write_json


def export_frame(
    predictions, reference, *, key, score, output_column, task="binary_classification"
):
    if task not in {"binary_classification", "regression"}:
        raise ValueError("Unsupported export task")
    if key == output_column or score == key:
        raise ValueError("key and score/output columns must differ")
    if not predictions.columns.is_unique or not reference.columns.is_unique:
        raise ValueError("duplicate columns")
    if key not in predictions or score not in predictions or key not in reference:
        raise ValueError("key or score column missing")
    for frame in (predictions, reference):
        if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("keys must be nonempty, unique and non-null")
    if len(predictions) != len(reference) or set(predictions[key]) != set(reference[key]):
        raise ValueError("prediction/reference key coverage differs")
    if pd.api.types.is_bool_dtype(predictions[score].dtype):
        raise ValueError("boolean flags are not probabilities")
    values = pd.to_numeric(predictions[score], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "invalid probabilities"
            if task == "binary_classification"
            else "invalid regression predictions"
        )
    if task == "binary_classification" and ((values < 0) | (values > 1)).any():
        raise ValueError("invalid probabilities")
    selected = pd.DataFrame({key: predictions[key].to_numpy(), output_column: values})
    result = reference[[key]].merge(selected, on=key, how="left", sort=False, validate="one_to_one")
    if result[key].tolist() != reference[key].tolist() or result[output_column].isna().any():
        raise ValueError("key alignment failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["binary_classification", "regression"], default="binary_classification"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--score", default="p_release")
    parser.add_argument("--output-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sidecar = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("immutable export already exists; choose another output path")
    before = {"predictions": sha256(args.predictions), "reference": sha256(args.reference)}
    if args.predictions.suffix.lower() == ".parquet":
        predictions = pd.read_parquet(args.predictions)
    else:
        predictions = pd.read_csv(args.predictions, dtype={args.key: "string"})
    predictions[args.key] = predictions[args.key].astype("string")
    reference = pd.read_csv(args.reference, dtype={args.key: "string"})
    if reference.columns.tolist() != [args.key, args.output_column]:
        raise ValueError("reference must have exactly the intended two-column header")
    result = export_frame(
        predictions,
        reference,
        key=args.key,
        score=args.score,
        output_column=args.output_column,
        task=args.task,
    )
    if before != {"predictions": sha256(args.predictions), "reference": sha256(args.reference)}:
        raise ValueError("export inputs changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False, float_format="%.17g")
    report = {
        "rows": len(result),
        "key": args.key,
        "score": args.score,
        "output_column": args.output_column,
        "input_sha256": before,
        "output_sha256": sha256(args.output),
    }
    if args.task == "regression":
        report.update(task="regression", prediction_kind="continuous")
    write_json(sidecar, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
