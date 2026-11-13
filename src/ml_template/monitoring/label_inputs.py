"""Version complete keyed outcome snapshots. Missing labels remain unknown."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.db import monitoring_registry as registry
from ml_template.monitoring.artifacts import digest, new_id, read_json, seal, verify, write_json
from ml_template.monitoring.monitor_inputs import load_batch


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("timestamps require explicit timezones")
    return result.tz_convert("UTC")


def validate_labels(frame, batch):
    key = batch.key
    expected = {key, "scoring_batch_id", "y_true", "observed_at", "eligible_at"}
    if frame.columns.duplicated().any() or set(frame.columns) != expected:
        raise ValueError(f"label columns must be exactly {sorted(expected)}")
    if (
        frame[key].isna().any()
        or frame[key].duplicated().any()
        or len(frame) != len(batch.predictions)
    ):
        raise ValueError("label snapshot must contain each scored key exactly once")
    expected_type = str if batch.contract["key_kind"] == "str" else (int, np.integer)
    if (
        not frame[key]
        .map(lambda v: isinstance(v, expected_type) and not isinstance(v, (bool, np.bool_)))
        .all()
    ):
        raise ValueError("label key type differs from scoring contract")
    if (
        set(frame[key]) != set(batch.predictions[key])
        or not frame.scoring_batch_id.eq(batch.batch_id).all()
    ):
        raise ValueError("label snapshot must cover exactly the specified scoring batch")
    out = frame.copy()
    present = out.y_true.notna()
    regression = getattr(batch, "task", "binary_classification") == "regression"
    if regression:
        from ml_template.monitoring.regression import validate_outcomes

        out["y_true"] = validate_outcomes(out.y_true)
    elif not out.loc[present, "y_true"].isin([0, 1]).all():
        raise ValueError("labels must be 0, 1, or missing")
    if not out.loc[present, "y_true"].map(lambda v: not isinstance(v, (bool, np.bool_))).all():
        raise ValueError("boolean labels are not supported")
    if not present.equals(out.observed_at.notna()):
        raise ValueError("observed_at is required exactly when a label is present")
    out["eligible_at"] = out.eligible_at.map(timestamp)
    out["observed_at"] = pd.to_datetime(
        out.observed_at.map(lambda v: timestamp(v) if pd.notna(v) else pd.NaT), utc=True
    )
    out["y_true"] = out.y_true.astype("Float64" if regression else "Int64")
    return out.set_index(key).loc[batch.predictions[key]].reset_index()


def import_labels(engine, config, *, output_root, check_only=False):
    batch = load_batch(engine, config.scoring_batch_id)
    from ml_template.monitoring.nn_monitoring_contract import context_for

    nn_context = context_for(batch, config)
    from ml_template.monitoring.regression import validate_options

    validate_options(batch, config)
    path = config.input_path.resolve()
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("labels require .csv or .parquet")
    before = digest(path)
    dtypes = {"scoring_batch_id": "string"}
    if batch.contract["key_kind"] == "str":
        dtypes[batch.key] = "string"
    frame = (
        pd.read_csv(path, dtype=dtypes) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    )
    normalised = validate_labels(frame, batch)
    if digest(path) != before:
        raise ValueError("label input changed during validation")
    if config.supersedes_label_import_id:
        prior = load_labels(engine, config.supersedes_label_import_id, batch)
        if prior["metadata"].get("target_unit") != config.target_unit:
            raise ValueError("corrected snapshots must preserve target units")
        if prior["metadata"]["outcome_definition"] != config.outcome_definition:
            raise ValueError("corrected snapshots must preserve the outcome definition")
    if check_only:
        return {
            "ready": True,
            "n_rows": len(normalised),
            "n_labels": int(normalised.y_true.notna().sum()),
            "files_written": False,
        }
    identity = new_id("labels")
    directory = Path(output_root).resolve() / identity
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / f"source{path.suffix.lower()}"
    shutil.copyfile(path, snapshot)
    if digest(snapshot) != before or digest(path) != before:
        raise ValueError("label input changed during snapshotting")
    normalised.to_parquet(directory / "labels.parquet", index=False)
    metadata = {
        **config.model_dump(mode="json"),
        "label_import_id": identity,
        "release_id": batch.release_id,
        "source_sha256": before,
        "scoring_hashes": batch.artifact_hashes,
        **({"nn_context": nn_context} if nn_context is not None else {}),
    }
    write_json(directory / "labels.json", metadata)
    manifest_hash = seal(directory)
    registry.register_object(
        engine,
        identity,
        "labels",
        batch.batch_id,
        batch.release_id,
        directory,
        manifest_hash,
        metadata,
    )
    return {
        "label_import_id": identity,
        "artifact_dir": str(directory),
        "n_labels": int(normalised.y_true.notna().sum()),
    }


def load_labels(engine, identity, batch):
    record = registry.get_object(engine, identity, "labels")
    directory = Path(record["artifact_dir"])
    verify(directory, record["manifest_sha256"])
    metadata = read_json(directory / "labels.json")
    if (
        metadata["label_import_id"] != identity
        or record["scoring_batch_id"] != batch.batch_id
        or metadata["scoring_batch_id"] != batch.batch_id
        or record["release_id"] != batch.release_id
    ):
        raise ValueError("label import belongs to a different scoring batch")
    if (
        metadata["release_id"] != batch.release_id
        or metadata["scoring_hashes"] != batch.artifact_hashes
    ):
        raise ValueError("label import prediction provenance mismatch")
    from ml_template.monitoring.nn_monitoring_contract import validate_saved_context

    validate_saved_context(metadata, batch, same_batch=True)
    frame = validate_labels(pd.read_parquet(directory / "labels.parquet"), batch)
    return {"metadata": metadata, "frame": frame}
