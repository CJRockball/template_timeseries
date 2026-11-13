"""Strict saved scoring-batch identity and exact keyed prediction coverage."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.explainability.artifacts import (
    fingerprint,
    read_json,
    safe_path,
    sha256,
    verify_files,
)


@dataclass(frozen=True)
class SavedBatch:
    directory: Path
    summary: dict
    binding: dict
    frame: pd.DataFrame
    predictions: pd.DataFrame
    contract: dict
    release_info: dict


def inspect_saved_batch(config):
    from ml_template.scoring.release_artifacts import verify

    release = Path(config.release_dir)
    manifest = verify(release, config.expected_release_id)
    info = read_json(safe_path(release, "release.json"))
    contract = read_json(safe_path(release, "contract.json"))
    directory = Path(config.scoring_dir)
    artifacts = read_json(safe_path(directory, "artifacts.json"))
    if set(artifacts) != {"files"}:
        raise ValueError("unsupported scoring artifact manifest")
    entries = artifacts["files"]
    snapshots = [name for name in ("input.csv", "input.parquet") if name in entries]
    if len(snapshots) != 1 or set(entries) != {snapshots[0], "predictions.parquet", "summary.json"}:
        raise ValueError("expected exactly one saved input, predictions and summary")
    verify_files(directory, entries)
    summary = read_json(safe_path(directory, "summary.json"))
    if summary.get("labels_loaded") is not False or type(summary.get("schema_version")) is not int:
        raise ValueError("scoring summary must affirm label-free schema version")
    expected = {
        "schema_version": 1,
        "scoring_batch_id": config.expected_scoring_batch_id,
        "release_id": config.expected_release_id,
        "content_sha256": manifest["content_sha256"],
        "parent_run_id": info["parent_run_id"],
        "calibration_id": info["calibration_id"],
        "release_variant": info["release_variant"],
        "policy": info["policy"],
        "labels_loaded": False,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"scoring/release identity mismatch: {name}")
    if type(summary.get("n_rows")) is not int or not 0 < summary["n_rows"] <= config.max_rows:
        raise ValueError("invalid scoring row count or max_rows exceeded")
    snapshot = snapshots[0]
    if summary["input_sha256"] != entries[snapshot]["sha256"]:
        raise ValueError("scoring input hash differs from manifest")
    policy = info["policy"]
    if summary.get("batch_definition") != policy.get("batch_definition"):
        raise ValueError("saved decision batch definition mismatch")
    if policy["kind"] == "alert_rate":
        if summary.get("batch_definition_acknowledged") is not True:
            raise ValueError("original alert-rate scoring lacks batch acknowledgement")
        if config.batch_definition != policy["batch_definition"]:
            raise ValueError("explicit original alert-rate batch definition required")
    elif config.batch_definition is not None and config.batch_definition != policy.get(
        "batch_definition"
    ):
        raise ValueError("batch definition differs from release")
    binding = {
        "release_id": manifest["release_id"],
        "release_content_sha256": manifest["content_sha256"],
        "scoring_batch_id": summary["scoring_batch_id"],
        "scoring_manifest_sha256": sha256(directory / "artifacts.json"),
        "scoring_files": entries,
        "snapshot_name": snapshot,
        "summary_sha256": entries["summary.json"]["sha256"],
    }
    binding["binding_sha256"] = fingerprint(binding)
    return directory, summary, binding, contract, info


def validate_predictions(frame, predictions, contract, info, summary):
    key = contract["key_col"]
    source_names = [item["name"] for item in contract["source_schema"]]
    if frame.columns.duplicated().any() or set(frame.columns) != {key, *source_names}:
        raise ValueError("snapshot source columns differ from frozen contract")
    keys = frame[key]
    if keys.isna().any() or keys.duplicated().any() or len(keys) != summary["n_rows"]:
        raise ValueError("snapshot entity keys/count invalid")
    kind = contract["key_kind"]
    if kind not in {"str", "int"}:
        raise ValueError("unsupported entity key kind")

    def valid(value):
        if kind == "str":
            return isinstance(value, str)
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

    if not keys.map(valid).all():
        raise ValueError("snapshot entity key type differs")
    expected = {key, "p_raw", "p_release", "release_id", "scoring_batch_id"}
    if info["calibration_id"] is not None:
        expected.add("p_calibrated")
    if info["policy"]["kind"] != "none":
        expected.add("policy_flag")
    if predictions.columns.duplicated().any() or set(predictions.columns) != expected:
        raise ValueError("unexpected scoring prediction schema")
    pkeys = predictions[key]
    if pkeys.isna().any() or pkeys.duplicated().any() or not pkeys.map(valid).all():
        raise ValueError("invalid prediction entity keys")
    if set(pkeys) != set(keys) or len(predictions) != len(frame):
        raise ValueError("predictions must cover every snapshot row exactly once")
    if not (predictions.release_id == summary["release_id"]).all():
        raise ValueError("prediction release identity mismatch")
    if not (predictions.scoring_batch_id == summary["scoring_batch_id"]).all():
        raise ValueError("prediction scoring-batch identity mismatch")
    aligned = predictions.set_index(key).loc[keys.to_numpy()].copy()
    aligned.index.name = key
    for name in ("p_raw", "p_release", "p_calibrated"):
        if name in aligned:
            if not pd.api.types.is_numeric_dtype(aligned[name].dtype) or pd.api.types.is_bool_dtype(
                aligned[name].dtype
            ):
                raise ValueError("probability columns must be numeric")
            values = aligned[name].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("invalid saved probabilities")
    selected = aligned[f"p_{info['release_variant']}"]
    if not np.array_equal(selected.to_numpy(), aligned.p_release.to_numpy()):
        raise ValueError("saved release probabilities differ from selected variant")
    if "policy_flag" in aligned:
        if aligned.policy_flag.isna().any() or not pd.api.types.is_bool_dtype(
            aligned.policy_flag.dtype
        ):
            raise ValueError("saved policy flags must be non-null booleans")
        flagged = int(aligned.policy_flag.sum())
        if summary.get("n_flagged") != flagged or summary.get("flagged_fraction") != flagged / len(
            aligned
        ):
            raise ValueError("saved policy flag summary differs")
    elif summary.get("n_flagged") is not None or summary.get("flagged_fraction") is not None:
        raise ValueError("unexpected policy flag summary")
    return aligned


def load_saved_batch(config):
    directory, summary, binding, contract, info = inspect_saved_batch(config)
    snapshot = directory / binding["snapshot_name"]
    if snapshot.suffix == ".csv":
        kwargs = {"dtype": {contract["key_col"]: "string"}} if contract["key_kind"] == "str" else {}
        frame = pd.read_csv(snapshot, **kwargs)
    else:
        frame = pd.read_parquet(snapshot)
    predictions = validate_predictions(
        frame, pd.read_parquet(directory / "predictions.parquet"), contract, info, summary
    )
    if inspect_saved_batch(config)[2] != binding:
        raise ValueError("scoring inputs changed while loading")
    return SavedBatch(directory, summary, binding, frame, predictions, contract, info)
