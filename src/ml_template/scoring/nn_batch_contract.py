"""Pure keyed batch/export contracts; no model or evaluation execution."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(root, name):
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if (
        not name
        or part.is_absolute()
        or ".." in part.parts
        or ("\\" in name)
        or (part.as_posix() != name)
    ):
        raise ValueError("Unsafe batch artifact path")
    path = root.joinpath(*part.parts)
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))
    ):
        raise ValueError("Unsafe batch artifact symlink")
    return path


def key_fingerprint(values):
    index = pd.Index(values)
    if index.empty or index.hasnans or (not index.is_unique):
        raise ValueError("Keys must be nonempty, unique and nonmissing")
    encoded = []
    kinds = set()
    for value in index:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Keys must be integers or strings")
        kinds.add(type(value))
        encoded.append(json.dumps([type(value).__name__, value]))
    if len(kinds) != 1:
        raise ValueError("Mixed key types")
    return hashlib.sha256(json.dumps(sorted(encoded), separators=(",", ":")).encode()).hexdigest()


def output_columns(info):
    if (
        info.get("task") != "regression"
        or info.get("prediction_kind") != "continuous"
        or info.get("model_type") != "nn"
    ):
        raise ValueError("NN continuous release required")
    result = ["y_pred_raw", "y_pred_release"]
    for item in info.get("interval_columns", {}).values():
        result.extend([item["lower"], item["upper"]])
    if len(result) != len(set(result)):
        raise ValueError("Duplicate frozen output columns")
    return result


def validate_export_mapping(columns, key, allowed):
    if not isinstance(columns, dict) or not columns or (not set(columns) <= set(allowed)):
        raise ValueError("Select only frozen numeric prediction columns")
    names = list(columns.values())
    if (
        any(not isinstance(n, str) or not n.strip() for n in names)
        or len(names) != len(set(names))
        or key in names
    ):
        raise ValueError("Output names must be unique, nonempty and not the key")


def export_frame(predictions, reference, *, key, columns, allowed):
    validate_export_mapping(columns, key, allowed)
    if (
        not predictions.columns.is_unique
        or not reference.columns.is_unique
        or key not in predictions
        or (key not in reference)
    ):
        raise ValueError("Invalid export schema")
    if key_fingerprint(predictions[key]) != key_fingerprint(reference[key]) or len(
        predictions
    ) != len(reference):
        raise ValueError("Exact reference key coverage required; intersections forbidden")
    selected = pd.DataFrame({key: predictions[key].to_numpy()})
    for source, destination in columns.items():
        a = predictions[source].to_numpy()
        if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("Prediction columns must be finite real values, not strings/flags")
        selected[destination] = a
    out = reference[[key]].merge(selected, on=key, how="left", sort=False, validate="one_to_one")
    if (
        out[key].tolist() != reference[key].tolist()
        or out[list(columns.values())].isna().any().any()
    ):
        raise ValueError("Export ordering/alignment failed")
    return out


def verify_batch(directory, expected_manifest=None):
    directory = Path(directory)
    path = safe(directory, "artifacts.json")
    if expected_manifest and sha256(path) != expected_manifest:
        raise ValueError("Scoring manifest changed")
    meta = json.loads(path.read_text())
    entries = meta["files"]
    if not {"predictions.parquet", "summary.json"} <= set(entries):
        raise ValueError("Incomplete scored batch")
    actual = set()
    for p in directory.rglob("*"):
        if p.is_symlink():
            raise ValueError("Batch symlinks forbidden")
        if p.is_file() and p.name != "artifacts.json":
            actual.add(p.relative_to(directory).as_posix())
    if actual != set(entries):
        raise ValueError("Missing/unlisted batch files")
    for name, item in entries.items():
        if sha256(safe(directory, name)) != item["sha256"]:
            raise ValueError("Scoring artifact changed: " + name)
    return json.loads(safe(directory, "summary.json").read_text())


def validate_predictions(frame, summary, info, key, release_id, batch_id):
    numeric = output_columns(info)
    if (
        set(frame) != {key, "release_id", "scoring_batch_id", *numeric}
        or not frame.columns.is_unique
    ):
        raise ValueError("Unexpected scored prediction schema")
    key_fingerprint(frame[key])
    if (
        len(frame) != summary["n_rows"]
        or not frame.release_id.eq(release_id).all()
        or (not frame.scoring_batch_id.eq(batch_id).all())
    ):
        raise ValueError("Prediction identity/support differs")
    if (
        summary.get("release_id") != release_id
        or summary.get("scoring_batch_id") != batch_id
        or summary.get("parent_run_id") != info["parent_run_id"]
        or (summary.get("calibration_id") != info["calibration_id"])
        or (summary.get("release_variant") != info["release_variant"])
        or (summary.get("prediction_columns") != info["prediction_columns"])
        or (
            summary.get("target_unit") is not None and summary["target_unit"] != info["target_unit"]
        )
        or (
            info.get("postprocessing") is not None
            and (
                summary.get("postprocessing") != info["postprocessing"]
                or summary.get("interval_columns") != info.get("interval_columns")
            )
        )
    ):
        raise ValueError("Scoring/release metadata binding differs")
    if (
        summary.get("task") != "regression"
        or summary.get("prediction_kind") != "continuous"
        or summary.get("labels_loaded") is not False
        or (summary.get("n_flagged") is not None)
    ):
        raise ValueError("Wrong scoring semantics")
    for name in numeric:
        values = frame[name].to_numpy()
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Nonfinite/non-real scored predictions")
    for item in info.get("interval_columns", {}).values():
        if (frame[item["lower"]] > frame[item["upper"]]).any():
            raise ValueError("Inverted intervals")
    if info.get("postprocessing") is None and (
        not np.array_equal(frame.y_pred_raw, frame.y_pred_release)
    ):
        raise ValueError("Uncorrected points differ")
    return frame
