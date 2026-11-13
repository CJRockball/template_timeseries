"""Resolve calibration inputs without loading holdout labels or feature matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_template.models.probability_calibrator import labels, probabilities


def align_development(oof, assignments, targets, key_col, target_col):
    for name, frame in [("OOF", oof), ("assignments", assignments), ("targets", targets)]:
        if frame[key_col].isna().any() or frame[key_col].duplicated().any():
            raise ValueError(f"{name}: keys must be unique and non-null")
    if not assignments["partition"].isin(["dev", "holdout"]).all():
        raise ValueError("unknown split partition")
    dev = assignments.loc[assignments["partition"].eq("dev"), [key_col, "fold"]]
    if dev.empty or dev["fold"].isna().any() or (dev["fold"] < 0).any():
        raise ValueError("invalid development folds")
    if set(oof[key_col]) != set(dev[key_col]):
        raise ValueError("OOF keys must cover exactly development rows; holdout is forbidden")
    if set(targets[key_col]) != set(dev[key_col]):
        raise ValueError("labels must cover exactly development rows")
    merged = oof[[key_col, "fold", "p_oof"]].merge(
        dev, on=key_col, validate="one_to_one", suffixes=("", "_stored")
    )
    if not merged["fold"].eq(merged["fold_stored"]).all():
        raise ValueError("OOF fold identities disagree with stored split")
    merged = merged.drop(columns="fold_stored").merge(
        targets[[key_col, target_col]], on=key_col, validate="one_to_one"
    )
    merged = merged.rename(columns={"p_oof": "p_raw", target_col: "y_true"})
    probabilities(merged["p_raw"])
    labels(merged["y_true"], len(merged))
    return merged.sort_values(key_col).reset_index(drop=True)


@dataclass(frozen=True)
class CalibrationInputs:
    frame: pd.DataFrame
    key_col: str
    provenance: dict


def load_calibration_inputs(engine, parent_run_id: str, *, silver_root=None, split_root=None):
    from ml_template.db import model_run_registry as registry
    from ml_template.paths import DATA_SILVER, DATA_SPLITS
    from ml_template.tracking import run_artifacts as artifacts
    from ml_template.training.run_config import RunConfig, config_hash
    from ml_template.training.split import compute_split_id

    parent = registry.get_run(engine, parent_run_id)
    if parent is None or parent["status"] != "succeeded":
        raise ValueError("parent run must exist and have succeeded")
    run_dir = artifacts.resolve_stored_path(parent["artifact_dir"])
    for name in ["config.json", "features.json", "model.ubj", "oof_predictions.parquet"]:
        artifacts.verify_artifact(run_dir, name)
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.task == "regression":
        raise ValueError(
            "Regression calibration/evaluation/release is not implemented by this training patch"
        )
    if config_hash(cfg) != parent["config_hash"]:
        raise ValueError("parent config hash disagrees with registry")
    for name in ["silver_dataset_version_id", "feature_set_id"]:
        if getattr(cfg.data, name) != parent[name]:
            raise ValueError(f"parent {name} disagrees with registry")
    feature_info = json.loads((run_dir / "features.json").read_text(encoding="utf-8"))
    if feature_info["split_id"] != parent["split_id"]:
        raise ValueError("parent feature metadata split mismatch")
    bound_files = ["config.json", "features.json", "model.ubj", "oof_predictions.parquet"]
    if cfg.fitted_features:
        artifacts.verify_artifact(run_dir, "model.transformer.joblib")
        bound_files.append("model.transformer.joblib")

    silver_dir = (
        Path(silver_root or DATA_SILVER)
        / cfg.data.dataset_name
        / cfg.data.silver_dataset_version_id
    )
    silver_manifest_path = silver_dir / "manifest.json"
    silver_manifest = json.loads(silver_manifest_path.read_text(encoding="utf-8"))
    if (
        not silver_manifest.get("validation_passed")
        or silver_manifest["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
    ):
        raise ValueError("invalid Silver version manifest")
    split_dir = Path(split_root or DATA_SPLITS) / parent["split_id"]
    split_manifest_path = split_dir / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    key_col, target_col = split_manifest["key_col"], split_manifest["target_col"]
    roles = silver_manifest["roles"]
    ids = roles.get("id", roles.get("ids", []))
    target_roles = roles.get("target", roles.get("targets", []))
    if key_col not in ids or target_roles != [target_col]:
        raise ValueError("Silver id/target roles disagree with stored split")
    if split_manifest["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id:
        raise ValueError("split belongs to a different Silver version")
    if (
        compute_split_id(cfg.data.silver_dataset_version_id, cfg.split, key_col, target_col)
        != parent["split_id"]
    ):
        raise ValueError("config split identity mismatch")
    assignments_path = split_dir / "assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    import hashlib

    observed = hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
    if observed != split_manifest["assignments_sha256"]:
        raise ValueError("split assignment hash mismatch")
    parquet_path = silver_dir / "data.parquet"
    silver_hash = artifacts.sha256_file(parquet_path)
    if silver_hash != silver_manifest["parquet_sha256"]:
        raise ValueError("Silver data hash mismatch")
    dev_keys = assignments.loc[assignments["partition"].eq("dev"), key_col].tolist()
    if not dev_keys:
        raise ValueError("empty development split")
    targets = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
        columns=[key_col, target_col],
        filters=[(key_col, "in", dev_keys)],
    )
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    frame = align_development(oof, assignments, targets, key_col, target_col)
    provenance = {
        "parent_run_id": parent_run_id,
        "artifact_dir": parent["artifact_dir"],
        "config_hash": parent["config_hash"],
        "split_id": parent["split_id"],
        "silver_dataset_version_id": cfg.data.silver_dataset_version_id,
        "feature_set_id": cfg.data.feature_set_id,
        "key_col": key_col,
        "target_col": target_col,
        "parent_holdout_evaluated_at": parent.get("holdout_evaluated_at"),
        "parent_files_sha256": {n: artifacts.sha256_file(run_dir / n) for n in bound_files},
        "silver_parquet_sha256": silver_hash,
        "silver_manifest_sha256": artifacts.sha256_file(silver_manifest_path),
        "split_manifest_sha256": artifacts.sha256_file(split_manifest_path),
        "split_assignments_sha256": artifacts.sha256_file(assignments_path),
    }
    return CalibrationInputs(frame, key_col, provenance)
