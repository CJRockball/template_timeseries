"""Metadata-only regression preflight: no target or feature Parquet reads."""

import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_template.evaluation.evaluation_artifacts import sha256
from ml_template.evaluation.evaluation_inputs import canonical_keys, digest, holdout_fingerprint


@dataclass
class PreparedRegression:
    spec: object
    run_cfg: object
    run_dir: Path
    assignments: pd.DataFrame
    hold_keys: list
    key_col: str
    target_col: str
    feature_names: list
    feature_info: dict
    parquet_path: Path
    provenance: dict
    identity_hash: str
    fingerprint: str
    roots: dict
    purpose: str = "evaluation"


def prepare_regression(engine, spec, *, silver_root=None, split_root=None, feature_root=None):
    """Existing evaluation/calibration/release boundary remains XGB-only in Package 01."""
    return _prepare_regression_inputs(
        engine, spec, silver_root=silver_root, split_root=split_root, feature_root=feature_root
    )


def prepare_nn_regression_predictor_inputs(
    engine, spec, *, silver_root=None, split_root=None, feature_root=None
):
    """NN metadata/predictor preparation only; no label loading or lifecycle authorisation."""
    return _prepare_regression_inputs(
        engine,
        spec,
        silver_root=silver_root,
        split_root=split_root,
        feature_root=feature_root,
        nn_predictor_only=True,
    )


def _prepare_regression_inputs(
    engine, spec, *, silver_root=None, split_root=None, feature_root=None, nn_predictor_only=False
):
    from ml_template.data.contract import load_dataset_contract
    from ml_template.db import model_run_registry as registry
    from ml_template.models.predictor_selection import describe_predictor
    from ml_template.paths import DATA_FEATURES, DATA_SILVER, DATA_SPLITS
    from ml_template.tracking import run_artifacts as artifacts
    from ml_template.training.run_config import RunConfig, config_hash, resolve_path
    from ml_template.training.split import compute_split_id

    if nn_predictor_only:
        # No ensure_tables/schema migrations in a read-only NN preparation.
        from sqlalchemy import text

        with engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT * FROM model_runs WHERE run_id=:run_id"),
                    {"run_id": spec.parent_run_id},
                )
                .mappings()
                .one_or_none()
            )
        parent = dict(row) if row is not None else None
    else:
        parent = registry.get_run(engine, spec.parent_run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("Parent run must exist and have succeeded")
    run_dir = artifacts.resolve_stored_path(parent["artifact_dir"])
    for name in ["config.json", "features.json", "model.ubj", "oof_predictions.parquet"]:
        artifacts.verify_artifact(run_dir, name)
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    if nn_predictor_only:
        if cfg.task != "regression" or cfg.model.type != "nn" or cfg.fitted_features:
            raise ValueError("Expected NN regression parent with embedded preprocessing")
        if (
            spec.calibration_id is not None
            or spec.release_variant != "raw"
            or spec.uncertainty.enabled
        ):
            raise ValueError("Package 01 supports raw NN predictor preparation only")
    elif cfg.task != "regression" or cfg.model.type != "xgb":
        raise ValueError("Expected XGB regression parent")
    from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

    validate_regression_specs(cfg.fitted_features)
    if config_hash(cfg) != parent["config_hash"]:
        raise ValueError("Parent config hash disagrees with registry")
    for name in ["silver_dataset_version_id", "feature_set_id"]:
        if getattr(cfg.data, name) != parent[name]:
            raise ValueError(f"Parent {name} disagrees with config")
    if parent["model_type"] != cfg.model.type or parent["seed"] != cfg.seed:
        raise ValueError("Parent model metadata disagrees with config")
    contract_path = resolve_path(cfg.data.dataset_config)
    raw_contract = yaml.safe_load(contract_path.read_text())
    if raw_contract.get("data", {}).get("task") != "regression":
        raise ValueError("Dataset contract task differs")
    contract = load_dataset_contract(contract_path)
    roots = {
        "silver_root": Path(silver_root or DATA_SILVER),
        "split_root": Path(split_root or DATA_SPLITS),
        "feature_root": Path(feature_root or DATA_FEATURES),
    }
    sd = roots["silver_root"] / cfg.data.dataset_name / cfg.data.silver_dataset_version_id
    sm_path, parquet = (sd / "manifest.json", sd / "data.parquet")
    sm = json.loads(sm_path.read_text())
    if (
        not sm.get("validation_passed")
        or sm["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
    ):
        raise ValueError("Invalid Silver manifest")
    roles = sm["roles"]
    ids = roles.get("id", roles.get("ids", []))
    targets = roles.get("target", roles.get("targets", []))
    features = roles.get("feature", roles.get("features", []))
    key, target = (contract.entity_id, contract.target)
    if key not in ids or targets != [target] or set(features) != set(contract.features):
        raise ValueError("Silver/contract roles disagree")
    if not features or len(features) != len(set(features)) or set(features) & {key, target}:
        raise ValueError("Invalid feature roles")
    info = json.loads((run_dir / "features.json").read_text())
    if (info.get("fitted_features") or []) != (cfg.fitted_features or []):
        raise ValueError("Saved fitted feature specification differs from run config")
    split_id = compute_split_id(cfg.data.silver_dataset_version_id, cfg.split, key, target)
    if parent["split_id"] != split_id or info["split_id"] != split_id:
        raise ValueError("Split identity differs")
    if info["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id:
        raise ValueError("Saved feature metadata Silver identity differs")
    sp = roots["split_root"] / split_id
    spm_path, assignments_path = (sp / "manifest.json", sp / "assignments.parquet")
    spm = json.loads(spm_path.read_text())
    if (
        spm["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
        or spm["key_col"] != key
        or spm["target_col"] != target
    ):
        raise ValueError("Split manifest roles/version disagree")
    if spm.get("split_id") != split_id:
        raise ValueError("Split manifest ID differs")
    assignments = pd.read_parquet(assignments_path)
    canonical_keys(assignments[key])
    from ml_template.training.regression_partitions import validate_assignments

    validate_assignments(cfg.split, assignments, spm)
    if assignments.fold.isna().any():
        raise ValueError("Invalid split partitions")
    fold_values = pd.to_numeric(assignments.fold, errors="raise")
    if not np.isfinite(fold_values).all() or (fold_values % 1 != 0).any():
        raise ValueError("Fold IDs must be finite integers")
    dev = assignments.partition.eq("dev")
    fold_ids = sorted(int(k) for k in assignments.loc[dev, "fold"].unique())
    if fold_ids != list(range(cfg.split.n_folds)) or not assignments.loc[~dev, "fold"].eq(-1).all():
        raise ValueError("Invalid development/holdout folds")
    hold_keys = (
        assignments.loc[assignments.partition.eq("holdout"), key]
        .sort_values(kind="stable")
        .tolist()
    )
    if not hold_keys or len(assignments) != spm["n_rows"] or len(hold_keys) != spm["n_holdout"]:
        raise ValueError("Split row counts differ")
    import hashlib

    if (
        hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
        != spm["assignments_sha256"]
    ):
        raise ValueError("Split assignment hash differs")
    if sha256(parquet) != sm["parquet_sha256"]:
        raise ValueError("Silver Parquet hash differs")
    resolved = spec.model_dump()
    for field, actual in [
        ("expected_silver_dataset_version_id", cfg.data.silver_dataset_version_id),
        ("expected_feature_set_id", cfg.data.feature_set_id),
        ("expected_split_id", split_id),
    ]:
        if resolved[field] is not None and resolved[field] != actual:
            raise ValueError(f"Unexpected {field}")
        resolved[field] = actual
    spec = type(spec).model_validate(resolved)
    nn_options = {}
    if nn_predictor_only:
        n_dev = int(dev.sum())
        if spec.predictor.kind == "final":
            counts = {"model.ubj": n_dev}
        else:
            counts = {
                f"fold_models/fold_{k}.ubj": int((dev & assignments.fold.ne(k)).sum())
                for k in fold_ids
            }
        nn_options["training_rows"] = counts
    descriptor = describe_predictor(
        cfg,
        run_dir,
        spec.predictor.kind,
        fold_ids=fold_ids,
        verify_artifact=artifacts.verify_artifact,
        **nn_options,
    )
    from ml_template.features.prepared_recipe import verify_plan

    preparation = verify_plan(cfg, features, info, key, len(assignments), roots["feature_root"])
    hashes = {
        "contract": sha256(contract_path),
        "config": sha256(run_dir / "config.json"),
        "features": sha256(run_dir / "features.json"),
        "silver_manifest": sha256(sm_path),
        "silver_parquet": sha256(parquet),
        "split_manifest": sha256(spm_path),
        "assignments": sha256(assignments_path),
        **descriptor["files_sha256"],
        **preparation["input_hashes"],
    }
    package = Path(__file__).resolve().parents[1]
    code_hashes = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))}
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "pydantic", "sqlalchemy"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    if nn_predictor_only:
        try:
            versions["torch"] = version("torch")
        except PackageNotFoundError:
            versions["torch"] = "not-installed"
    fingerprint = holdout_fingerprint(cfg.data.silver_dataset_version_id, key, hold_keys)
    provenance = {
        "task": "regression",
        "predictor": descriptor,
        "input_hashes": hashes,
        "code_hashes": code_hashes,
        "versions": versions,
        "python": sys.version,
        "holdout_fingerprint": fingerprint,
        "limitations": [
            "Stateless replay and target-free fold-fitted transforms supported; bootstrap intervals unsupported"
        ],
        "feature_preparation": preparation,
    }
    if spec.calibration_id is not None:
        from ml_template.evaluation.regression_postprocessing import attach

        attach(engine, spec, provenance)
    identity = digest({"spec": spec.model_dump(mode="json"), "provenance": provenance})
    return PreparedRegression(
        spec,
        cfg,
        run_dir,
        assignments,
        hold_keys,
        key,
        target,
        info["features"],
        info,
        parquet,
        provenance,
        identity,
        fingerprint,
        roots,
        purpose="predictor_only" if nn_predictor_only else "evaluation",
    )


def load_holdout(prepared):
    """Load only the already-approved holdout after access is recorded."""
    from ml_template.features.prepared_recipe import load_partition

    if getattr(prepared, "purpose", "evaluation") != "evaluation":
        raise ValueError("Predictor-only preparation does not authorise holdout loading")
    X, y = load_partition(prepared, prepared.hold_keys)
    return X, y.to_numpy(dtype=float)
