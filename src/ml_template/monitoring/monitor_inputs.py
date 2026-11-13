"""Verify registered score artifacts without loading a model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.db.monitoring_registry import rows
from ml_template.monitoring.artifacts import digest, read_json, safe_path, verify


@dataclass
class BatchView:
    record: dict
    summary: dict
    contract: dict
    source: pd.DataFrame
    predictions: pd.DataFrame
    artifact_hashes: dict

    @property
    def task(self):
        return self.summary.get("task", "binary_classification")

    @property
    def key(self):
        return self.contract["key_col"]

    @property
    def release_id(self):
        return self.record["release_id"]

    @property
    def batch_id(self):
        return self.record["scoring_batch_id"]


def load_batch(engine, batch_id):
    from ml_template.scoring.input_contract import validate_batch
    from ml_template.scoring.release_artifacts import verify as verify_release

    found = rows(engine, "SELECT * FROM scoring_batches WHERE scoring_batch_id=?", (batch_id,))
    if len(found) != 1 or found[0]["status"] != "succeeded":
        raise ValueError("an explicit successful scoring batch is required")
    record = found[0]
    releases = rows(
        engine, "SELECT * FROM model_releases WHERE release_id=?", (record["release_id"],)
    )
    if len(releases) != 1:
        raise ValueError("scoring release is not registered")
    release = releases[0]
    release_dir = Path(release["artifact_dir"])
    manifest = verify_release(release_dir, record["release_id"])
    if manifest["content_sha256"] != release["content_sha256"]:
        raise ValueError("registered release digest mismatch")
    contract = read_json(release_dir / "contract.json")
    info = read_json(release_dir / "release.json")
    directory = Path(record["artifact_dir"])
    entries = verify(directory)
    if not {"summary.json", "predictions.parquet"} <= entries.keys():
        raise ValueError("missing scoring artifacts")
    snapshots = [name for name in entries if name in {"input.csv", "input.parquet"}]
    if len(snapshots) != 1:
        raise ValueError("exactly one canonical input snapshot is required")
    summary = read_json(safe_path(directory, "summary.json"))
    task = info.get("task", "binary_classification")
    if task not in {"binary_classification", "regression"}:
        raise ValueError("unsupported monitoring task")
    regression = task == "regression"
    if (
        summary.get("schema_version") != (2 if regression else 1)
        or summary.get("task", "binary_classification") != task
    ):
        raise ValueError("unsupported score summary task/schema")
    for name in ("scoring_batch_id", "release_id", "n_rows", "n_flagged", "input_sha256"):
        if summary.get(name) != record[name]:
            raise ValueError(f"summary/registry mismatch: {name}")
    if (
        summary.get("content_sha256") != manifest["content_sha256"]
        or summary.get("policy") != info["policy"]
    ):
        raise ValueError("summary/release mismatch")
    snapshot = safe_path(directory, snapshots[0])
    if digest(snapshot) != record["input_sha256"]:
        raise ValueError("input hash differs from registry")
    key = contract["key_col"]
    kwargs = {"dtype": {key: "string"}} if contract["key_kind"] == "str" else {}
    source = (
        pd.read_csv(snapshot, **kwargs) if snapshot.suffix == ".csv" else pd.read_parquet(snapshot)
    )
    validate_batch(source, contract)
    predictions = pd.read_parquet(directory / "predictions.parquet")
    score_columns = {"y_pred_raw", "y_pred_release"} if regression else {"p_raw", "p_release"}
    required = {key, "release_id", "scoring_batch_id", *score_columns}
    if predictions.columns.duplicated().any() or not required <= set(predictions.columns):
        raise ValueError("invalid prediction columns")
    if len(predictions) != len(source) or len(source) != record["n_rows"]:
        raise ValueError("prediction coverage differs from input")
    keys = predictions[key]
    if keys.isna().any() or keys.duplicated().any():
        raise ValueError("invalid prediction keys")
    expected_type = str if contract["key_kind"] == "str" else (int, np.integer)
    if not keys.map(
        lambda v: isinstance(v, expected_type) and not isinstance(v, (bool, np.bool_))
    ).all():
        raise ValueError("prediction key type differs from contract")
    if set(keys) != set(source[key]):
        raise ValueError("prediction keys do not cover the scoring input")
    for column, expected in (("release_id", record["release_id"]), ("scoring_batch_id", batch_id)):
        if not predictions[column].eq(expected).all():
            raise ValueError(f"prediction identity mismatch: {column}")
    if regression:
        from ml_template.monitoring.regression import validate_predictions

        validate_predictions(predictions, summary, info)
        if info.get("model_type") == "nn":
            from ml_template.monitoring.nn_monitoring_contract import release_binding

            summary = release_binding(
                summary,
                info,
                manifest,
                record,
                release_dir,
                entries,
                digest(directory / "artifacts.json"),
            )
        predictions = predictions.set_index(key).loc[source[key]].reset_index()
        hashes = {name: item["sha256"] for name, item in entries.items()}
        hashes["artifacts.json"] = digest(directory / "artifacts.json")
        return BatchView(
            record, summary, contract, source.reset_index(drop=True), predictions, hashes
        )
    for column in ("p_raw", "p_calibrated", "p_release"):
        if column in predictions:
            values = predictions[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"invalid probability: {column}")
    selected = f"p_{summary['release_variant']}"
    if selected not in predictions or not np.array_equal(
        predictions.p_release, predictions[selected]
    ):
        raise ValueError("release probability differs from selected variant")
    has_policy = info["policy"]["kind"] != "none"
    if has_policy != ("policy_flag" in predictions):
        raise ValueError("policy flag presence differs from release")
    if has_policy:
        flags = predictions.policy_flag
        if flags.isna().any() or not flags.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("policy flags must be saved boolean decisions")
        if int(flags.sum()) != record["n_flagged"]:
            raise ValueError("saved flagged count mismatch")
    predictions = predictions.set_index(key).loc[source[key]].reset_index()
    hashes = {name: item["sha256"] for name, item in entries.items()}
    hashes["artifacts.json"] = digest(directory / "artifacts.json")
    return BatchView(record, summary, contract, source.reset_index(drop=True), predictions, hashes)
