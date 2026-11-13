"""Preflight reuses development-only calibration input integrity checks; no holdout labels."""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ml_template.evaluation.evaluation_config import EvaluationConfig


def digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def canonical_keys(values):
    s = pd.Series(values)
    if s.empty or s.isna().any() or s.duplicated().any():
        raise ValueError("keys must be nonempty, unique and non-null")
    result = []
    for v in s:
        v = v.item() if hasattr(v, "item") else v
        if not isinstance(v, (str, int)) or isinstance(v, bool):
            raise ValueError("keys must be integer or string")
        result.append(json.dumps([type(v).__name__, v]))
    return sorted(result)


def holdout_fingerprint(silver_id, key_col, keys):
    return digest({"silver": silver_id, "key": key_col, "members": canonical_keys(keys)})


@dataclass
class PreparedEvaluation:
    spec: object
    run_cfg: object
    run_dir: Path
    assignments: object
    hold_keys: list
    key_col: str
    target_col: str
    feature_names: list
    provenance: dict
    identity_hash: str
    fingerprint: str
    calibrator: object
    calibration_dir: object
    roots: dict


def prepare_evaluation(engine, spec, *, silver_root=None, split_root=None, feature_root=None):
    from ml_template.data.contract import load_dataset_contract
    from ml_template.db import calibration_registry, model_run_registry
    from ml_template.models.calibration_artifacts import load_calibrator_for_run
    from ml_template.paths import DATA_FEATURES, DATA_SILVER, DATA_SPLITS
    from ml_template.tracking import run_artifacts as a
    from ml_template.training.calibration_inputs import load_calibration_inputs
    from ml_template.training.run_config import RunConfig, resolve_path

    bound = load_calibration_inputs(
        engine, spec.parent_run_id, silver_root=silver_root, split_root=split_root
    )
    provenance = dict(bound.provenance)
    parent = model_run_registry.get_run(engine, spec.parent_run_id)
    run_dir = a.resolve_stored_path(parent["artifact_dir"])
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    contract_path = resolve_path(cfg.data.dataset_config)
    contract = load_dataset_contract(contract_path)
    key, target = provenance["key_col"], provenance["target_col"]
    if contract.entity_id != key or contract.target != target:
        raise ValueError("dataset contract identity changed")
    for expected, actual in [
        (spec.expected_silver_dataset_version_id, cfg.data.silver_dataset_version_id),
        (spec.expected_feature_set_id, cfg.data.feature_set_id),
        (spec.expected_split_id, parent["split_id"]),
    ]:
        if expected is not None and expected != actual:
            raise ValueError("unexpected dataset, feature-set or split identity")
    roots = {
        "silver_root": Path(silver_root or DATA_SILVER),
        "split_root": Path(split_root or DATA_SPLITS),
        "feature_root": Path(feature_root or DATA_FEATURES),
    }
    assignments = pd.read_parquet(roots["split_root"] / parent["split_id"] / "assignments.parquet")
    canonical_keys(assignments[key])
    keys = (
        assignments.loc[assignments.partition.eq("holdout"), key]
        .sort_values(kind="stable")
        .tolist()
    )
    fingerprint = holdout_fingerprint(cfg.data.silver_dataset_version_id, key, keys)
    info = json.loads((run_dir / "features.json").read_text())
    hashes = {"contract": a.sha256_file(contract_path)}
    if cfg.data.feature_set_id:
        fr = roots["feature_root"]
        path = fr / "_sets" / f"{cfg.data.feature_set_id}.json"
        fs = json.loads(path.read_text())
        identity = {
            "silver": cfg.data.silver_dataset_version_id,
            "blocks": fs["blocks"],
            "columns": fs["columns"],
            "parent": fs["parent_feature_set_id"],
        }
        expected = (
            "fs_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        )
        if (
            expected != cfg.data.feature_set_id
            or fs["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
            or fs["columns"] != info["features"]
        ):
            raise ValueError("feature-set identity/content changed")
        hashes["feature_set"] = a.sha256_file(path)
        for block in fs["blocks"]:
            bd = fr / block
            bm = json.loads((bd / "manifest.json").read_text())
            if (
                bm["key_col"] != key
                or bm["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
                or a.sha256_file(bd / "data.parquet") != bm["parquet_sha256"]
            ):
                raise ValueError("feature block identity/hash mismatch")
            for n in ["manifest.json", "data.parquet"]:
                hashes[f"block/{block}/{n}"] = a.sha256_file(bd / n)
    cal, caldir = None, None
    if spec.calibration_id:
        cal = load_calibrator_for_run(engine, spec.calibration_id, spec.parent_run_id)
        record = calibration_registry.get_calibration(engine, spec.calibration_id)
        caldir = a.resolve_stored_path(record["artifact_dir"])
        saved = json.loads((caldir / "provenance.json").read_text())
        for name in [
            "silver_dataset_version_id",
            "feature_set_id",
            "split_id",
            "silver_parquet_sha256",
            "silver_manifest_sha256",
            "split_manifest_sha256",
            "split_assignments_sha256",
        ]:
            if saved[name] != provenance[name]:
                raise ValueError(f"calibration provenance mismatch: {name}")
        for n in [
            "calibrator.json",
            "config.json",
            "provenance.json",
            "metrics.csv",
            "diagnostic_notes.json",
        ]:
            if (caldir / n).exists():
                a.verify_artifact(caldir, n)
                hashes[f"calibration/{n}"] = a.sha256_file(caldir / n)
    spec = EvaluationConfig.model_validate(
        {
            **spec.model_dump(),
            "expected_silver_dataset_version_id": cfg.data.silver_dataset_version_id,
            "expected_feature_set_id": cfg.data.feature_set_id,
            "expected_split_id": parent["split_id"],
        }
    )
    if spec.policy.selection is not None:
        from ml_template.evaluation.policy_selection import validate_policy_selection

        selection = validate_policy_selection(spec, provenance, caldir)
        provenance["policy_selection"] = selection
        hashes["policy_selection"] = spec.policy.selection.artifact_sha256
    for s in spec.slices:
        if s.column not in info["features"] or s.column in [key, target]:
            raise ValueError("slices must use saved input features")
    versions = {}
    for n in [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "xgboost",
        "torch",
        "pyarrow",
        "pydantic",
        "sqlalchemy",
        "plotly",
        "PyYAML",
        "joblib",
    ]:
        try:
            versions[n] = version(n)
        except PackageNotFoundError:
            versions[n] = "not-installed"
    package_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(p.relative_to(package_root)): a.sha256_file(p)
        for folder in [
            "evaluation",
            "models",
            "training",
            "features",
            "preprocessing",
            "data",
            "tracking",
            "db",
        ]
        for p in sorted((package_root / folder).rglob("*.py"))
    }
    # The mutable first-access timestamp is audit context, not predictor identity.
    provenance.pop("parent_holdout_evaluated_at", None)
    provenance.update(
        input_hashes=hashes,
        code_hashes=code_hashes,
        versions=versions,
        python=a.environment_info()["python"],
        holdout_fingerprint=fingerprint,
        development_metrics=[
            r
            for r in model_run_registry.get_metrics(engine, spec.parent_run_id, split="cv")
            if r["fold"] == -1
        ],
    )
    from ml_template.models.predictor_selection import describe_predictor

    stored_fold_ids = sorted(
        int(value) for value in assignments.loc[assignments.partition.eq("dev"), "fold"].unique()
    )
    descriptor = describe_predictor(
        cfg,
        run_dir,
        spec.predictor.kind,
        fold_ids=stored_fold_ids,
        verify_artifact=a.verify_artifact,
    )
    provenance["predictor"] = descriptor
    for artifact_name, artifact_hash in descriptor["files_sha256"].items():
        provenance["input_hashes"][f"predictor/{artifact_name}"] = artifact_hash
    identity_hash = digest({"spec": spec.model_dump(mode="json"), "provenance": provenance})
    return PreparedEvaluation(
        spec,
        cfg,
        run_dir,
        assignments,
        keys,
        key,
        target,
        info["features"],
        provenance,
        identity_hash,
        fingerprint,
        cal,
        caldir,
        roots,
    )


def exposure_warnings(engine, prepared):
    from ml_template.db import evaluation_registry, model_run_registry

    silver = prepared.spec.expected_silver_dataset_version_id
    warnings = [
        {"source": "evaluation_registry", **r}
        for r in evaluation_registry.prior_exposures(engine, silver)
    ]
    warnings.extend(
        {
            "source": "legacy_or_parent_timestamp",
            "run_id": r["run_id"],
            "accessed_at": r["holdout_evaluated_at"],
        }
        for r in model_run_registry.list_runs(engine)
        if r["silver_dataset_version_id"] == silver and r.get("holdout_evaluated_at")
    )
    return warnings


# calibrated-policy update v1
