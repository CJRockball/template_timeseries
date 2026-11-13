"""Optional model-agnostic residual-only analysis of an immutable successful run."""

import argparse
import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from ml_template.dashboard import regression_data as saved
from ml_template.evaluation.regression_diagnostics import compute_diagnostics, write_diagnostics
from ml_template.evaluation.residual_artifacts import KIND, augment_summary, check_keys


class ResidualConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: StrictBool = False
    run_id: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    error_sample_size: int = Field(25, ge=0, strict=True)
    feature_columns: list[str] = Field(default_factory=list)
    database: str = "data/project.db"
    output_root: str | None = None
    silver_root: str | None = None
    split_root: str | None = None

    @field_validator("target_unit")
    @classmethod
    def unit(cls, value):
        if not value.strip() or value.strip() in {"target_units", "target units"}:
            raise ValueError("Declare the actual target unit")
        return value.strip()

    @field_validator("feature_columns")
    @classmethod
    def columns(cls, values):
        if len(values) != len(set(values)) or any(not v for v in values):
            raise ValueError("Feature names must be unique and nonempty")
        return values


def _path(value, default=None):
    from ml_template.paths import ROOT

    path = Path(default if value is None else value).expanduser()
    return path if path.is_absolute() else ROOT / path


def split_identity(silver, policy, key, target):
    return (
        "split_"
        + hashlib.sha256(
            json.dumps(
                {"silver": silver, "policy": policy, "key": key, "target": target}, sort_keys=True
            ).encode()
        ).hexdigest()[:12]
    )


def validate_inventory(assignments, oof, key, n_folds, manifest):
    check_keys(assignments[key])
    check_keys(oof[key])
    if (
        assignments.columns.duplicated().any()
        or oof.columns.duplicated().any()
        or set(oof.columns) != {key, "fold", "y_pred_oof"}
    ):
        raise ValueError("Ambiguous assignment/OOF schema")
    if not assignments.partition.isin(["dev", "holdout", "calibration", "recalibration"]).all():
        raise ValueError("Unknown split partition")
    for frame in (assignments, oof):
        dtype = frame["fold"].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError("Fold IDs must be real numeric integers, not booleans/objects")
        numeric = frame["fold"].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError("Fold IDs must be finite integers")
    dev = assignments.loc[assignments.partition.eq("dev"), [key, "fold"]]
    if (
        set(dev.fold) != set(range(n_folds))
        or dev.fold.isna().any()
        or (dev.fold % 1 != 0).any()
        or not assignments.loc[~assignments.partition.eq("dev"), "fold"].eq(-1).all()
    ):
        raise ValueError("Invalid stored fold roles")
    if (
        len(assignments) != manifest["n_rows"]
        or int(assignments.partition.eq("holdout").sum()) != manifest["n_holdout"]
    ):
        raise ValueError("Split support differs")
    if manifest["fold_sizes"] != {
        str(int(k)): int(v) for k, v in dev.fold.value_counts().sort_index().items()
    }:
        raise ValueError("Split fold sizes differ")
    for role in ("calibration", "recalibration"):
        if "n_" + role in manifest and manifest["n_" + role] != int(
            assignments.partition.eq(role).sum()
        ):
            raise ValueError("Reserved role support differs")
    if (
        hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
        != manifest["assignments_sha256"]
    ):
        raise ValueError("Stored assignment checksum differs")
    if set(oof[key]) != set(dev[key]):
        raise ValueError("OOF must cover EXACTLY development keys; non-development forbidden")
    a = dev.set_index(key).sort_index()
    b = oof.set_index(key).sort_index()
    if not np.array_equal(a.fold, b.fold):
        raise ValueError("OOF/stored fold identity differs")
    values = b.y_pred_oof
    if (
        not pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_complex_dtype(values)
        or not np.isfinite(values.to_numpy(dtype=float)).all()
    ):
        raise ValueError("OOF predictions must be finite real numbers")
    return dev[key].tolist()


def prepare(conn, cfg):
    from ml_template.data.contract import load_dataset_contract
    from ml_template.paths import DATA_SILVER, DATA_SPLITS

    ctx = saved.context(conn, cfg.run_id)
    raw = ctx["config"]
    data = raw["data"]
    contract_path = _path(data["dataset_config"]).resolve()
    contract_hash = saved.sha256(contract_path)
    contract_raw = yaml.safe_load(contract_path.read_text())
    if contract_raw.get("data", {}).get("task") != "regression":
        raise ValueError("Expected a regression dataset contract")
    contract = load_dataset_contract(contract_path)
    key, target = contract.entity_id, contract.target
    if not set(cfg.feature_columns) <= set(contract.features) or set(cfg.feature_columns) & {
        key,
        target,
    }:
        raise ValueError("Select only explicitly declared raw Silver features for examples")
    sr = _path(cfg.silver_root, DATA_SILVER)
    sd = saved.safe(sr, data["dataset_name"] + "/" + data["silver_dataset_version_id"])
    sm_path = saved.safe(sd, "manifest.json")
    parquet = saved.safe(sd, "data.parquet")
    sm_hash = saved.sha256(sm_path)
    sm = saved.read_json(sm_path)
    roles = sm["roles"]
    if (
        not sm.get("validation_passed")
        or sm["silver_dataset_version_id"] != data["silver_dataset_version_id"]
        or sm["config_sha256"] != contract_hash
        or key not in roles.get("id", roles.get("ids", []))
        or roles.get("target", roles.get("targets", [])) != [target]
        or set(roles.get("feature", roles.get("features", []))) != set(contract.features)
    ):
        raise ValueError("Silver/contract identity or roles differ")
    if saved.sha256(parquet) != sm["parquet_sha256"]:
        raise ValueError("Silver bytes differ BEFORE target reading")
    policy = dict(raw["split"])
    if policy.get("cv_seed") is None:
        policy.pop("cv_seed", None)
    for role in ("calibration", "recalibration"):
        if not policy.get(role + "_fraction", 0):
            policy.pop(role + "_fraction", None)
    if policy.get("method") != "random":
        raise ValueError("Regression analysis requires the original random split")
    split_id = split_identity(data["silver_dataset_version_id"], policy, key, target)
    if split_id != ctx["header"]["split_id"]:
        raise ValueError("Saved configuration/stored split identity differs")
    sp = saved.safe(_path(cfg.split_root, DATA_SPLITS), split_id)
    spm_path = saved.safe(sp, "manifest.json")
    assignments_path = saved.safe(sp, "assignments.parquet")
    spm_hash = saved.sha256(spm_path)
    assignment_hash = saved.sha256(assignments_path)
    spm = saved.read_json(spm_path)
    if (
        spm["split_id"] != split_id
        or spm["silver_dataset_version_id"] != data["silver_dataset_version_id"]
        or spm["key_col"] != key
        or spm["target_col"] != target
        or spm["policy"] != policy
    ):
        raise ValueError("Split manifest binding differs")
    saved.verify(ctx["root"], ctx["entries"], ["oof_predictions.parquet"])
    assignments = pd.read_parquet(assignments_path)
    oof = pd.read_parquet(saved.safe(ctx["root"], "oof_predictions.parquet"))
    keys = validate_inventory(assignments, oof, key, raw["split"]["n_folds"], spm)
    for role in ("calibration", "recalibration"):
        present = bool(assignments.partition.eq(role).any())
        if present != (policy.get(role + "_fraction", 0) > 0):
            raise ValueError("Reserved population differs from declared split policy")
    paths = {
        contract_path: contract_hash,
        sm_path: sm_hash,
        parquet: sm["parquet_sha256"],
        spm_path: spm_hash,
        assignments_path: assignment_hash,
        ctx["root"] / "artifact_manifest.json": saved.sha256(
            ctx["root"] / "artifact_manifest.json"
        ),
    }
    for name in ("config.json", "features.json", "oof_predictions.parquet"):
        paths[saved.safe(ctx["root"], name)] = ctx["entries"][name]["sha256"]
    verify_sources(paths)
    return {
        "ctx": ctx,
        "contract": contract,
        "paths": paths,
        "parquet": parquet,
        "assignments": assignments,
        "oof": oof,
        "keys": keys,
        "policy": policy,
    }


def verify_sources(paths):
    for path, expected in paths.items():
        if saved.sha256(path) != expected:
            raise ValueError("Analysis input changed: " + str(path))


def run_residual_analysis(conn, cfg, *, check_only=True):
    if not cfg.enabled:
        if check_only:
            return {"ready": False, "enabled": False, "files_written": False}
        raise ValueError("Residual analysis is disabled")
    p = prepare(conn, cfg)
    status = {
        "ready": True,
        "enabled": True,
        "parent_run_id": cfg.run_id,
        "analysis_kind": KIND,
        "n_development": len(p["keys"]),
        "models_loaded": False,
        "predictions_computed": False,
        "holdout_labels_loaded": False,
        "development_labels_loaded": False,
        "registry_rows_written": False,
        "parent_artifacts_modified": False,
        "files_written": False,
    }
    if check_only:
        return status
    from ml_template.evaluation.evaluation_artifacts import manifest, write_json
    from ml_template.paths import ARTIFACTS_RUNS, DATA_SILVER, DATA_SPLITS

    output = _path(cfg.output_root, saved.analysis_root()).resolve()
    parent = p["ctx"]["root"].resolve()
    forbidden = [
        parent,
        Path(ARTIFACTS_RUNS).resolve(),
        _path(cfg.silver_root, DATA_SILVER).resolve(),
        _path(cfg.split_root, DATA_SPLITS).resolve(),
    ]
    if any(output.is_relative_to(path) for path in forbidden):
        raise ValueError("Analysis output must be outside run/Silver/split directories")
    if any(
        (q / name).is_file()
        for q in [output, *output.parents]
        for name in ("artifact_manifest.json", "manifest.json")
    ):
        raise ValueError("Cannot write inside a sealed artifact tree")
    identity = "analysis_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:10]
    directory = output / identity
    directory.mkdir(parents=True, exist_ok=False)
    state = {**status, "analysis_id": identity, "artifact_dir": str(directory), "status": "running"}
    write_json(directory / "analysis.json", state)
    try:
        verify_sources(p["paths"])
        key, target = p["contract"].entity_id, p["contract"].target
        # Target reads select only stored development keys; no model or recipe replay.
        frame = pd.read_parquet(
            p["parquet"],
            engine="pyarrow",
            columns=[key, target, *cfg.feature_columns],
            filters=[(key, "in", p["keys"])],
        )
        check_keys(frame[key])
        if set(frame[key]) != set(p["keys"]):
            raise ValueError("Development target coverage differs")
        frame = frame.set_index(key).loc[p["keys"]]
        y = frame[target]
        if (
            not pd.api.types.is_numeric_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_complex_dtype(y)
            or not np.isfinite(y.to_numpy(dtype=float)).all()
        ):
            raise ValueError("Development targets must be finite real numbers")
        y = y.astype(float)
        oof = p["oof"].assign(y_pred_oof=p["oof"]["y_pred_oof"].astype(float))
        result = compute_diagnostics(
            oof,
            frame[cfg.feature_columns],
            y,
            p["assignments"],
            key,
            permutation=False,
            error_sample_size=cfg.error_sample_size,
            target_unit=cfg.target_unit,
        )
        augment_summary(result)
        verify_sources(p["paths"])
        write_diagnostics(result, directory, plots=True)
        folded = pd.DataFrame(result.summary["fold_summaries"]).drop(columns="residual_quantiles")
        folded.to_csv(directory / "fold_summary.csv", index=False)
        report_path = directory / "report.html"
        with report_path.open("a", encoding="utf-8") as stream:
            stream.write(
                "<h2>Per-fold OOF residual summaries</h2>"
                + folded.to_html(index=False, escape=True)
            )
        input_hashes = {
            "config": p["ctx"]["entries"]["config.json"]["sha256"],
            "features": p["ctx"]["entries"]["features.json"]["sha256"],
            "oof": p["ctx"]["entries"]["oof_predictions.parquet"]["sha256"],
        }
        write_json(
            directory / "provenance.json",
            {
                "schema_version": 1,
                "analysis_kind": KIND,
                "parent_run_id": cfg.run_id,
                "prediction_source": "original_saved_oof",
                "input_provenance": {"input_hashes": input_hashes},
                "verified_sources": {str(path): digest for path, digest in p["paths"].items()},
                "models_loaded": False,
                "predictions_computed": False,
                "holdout_labels_loaded": False,
                "registry_rows_written": False,
                "feature_columns_for_examples": cfg.feature_columns,
                "options": cfg.model_dump(mode="json"),
            },
        )
        verify_sources(p["paths"])
        state.update(
            status="succeeded",
            development_labels_loaded=True,
            files_written=True,
            report_path=str(directory / "report.html"),
        )
        write_json(directory / "analysis.json", state)
        manifest(directory)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "analysis.json", state)
        raise
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    cfg = ResidualConfig.model_validate(yaml.safe_load(args.config.read_text()))
    database = _path(cfg.database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        result = run_residual_analysis(connection, cfg, check_only=args.check)
    finally:
        connection.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
