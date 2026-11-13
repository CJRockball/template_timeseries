"""Build the Silver dataset (Parquet) from one Bronze dataset version.

Silver = canonical names, declared dtypes, roles, validation. No imputation,
scaling, encoding, outlier removal, or derived/physical features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml_template.data.canonicalize import canonicalize_frame
from ml_template.data.extract import extract_bronze
from ml_template.db import data_run_registry
from ml_template.db.connection import get_engine
from ml_template.tracking.pipeline_logging import setup_logging

logger = logging.getLogger("ml_template.scripts.silver")

TEXT_DTYPES = {"object", "category", "string"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_canonicalize(
    dataset_version_id: str,
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    engine = get_engine()
    raw = extract_bronze(
        dataset_version_id=dataset_version_id,
        engine=engine,
    )

    contract = data_config["silver_eda_contract"]
    canonical_names = contract["canonical_names"]

    df = canonicalize_frame(
        raw,
        contract,
        columns=list(contract["canonical_dtypes"]),
        passthrough=("dataset_version_id", "source_row_number"),
    )

    # Retain source ordering, including lineage columns.
    ordered_columns = [
        canonical_names.get(col, col)
        for col in raw.columns
        if canonical_names.get(col, col) in df.columns
    ]
    df = df.loc[:, ordered_columns]

    roles = data_config["bronze_source_contract"]["roles"]
    role_map = {
        role: [canonical_names.get(col, col) for col in columns] for role, columns in roles.items()
    }

    targets = role_map.get("target", [])
    if len(targets) != 1:
        raise ValueError("Expected exactly one target column")

    target = targets[0]
    if df[target].isna().any():
        raise ValueError(f"{target}: missing target labels are not allowed")

    logger.info(
        "Loaded %d rows, %d columns for dataset_version_id=%s",
        len(df),
        df.shape[1],
        dataset_version_id,
    )
    return df, role_map


def get_silver_version_id(dataset_version_id: str, config_path: Path) -> str:
    return f"silver_{dataset_version_id}_{sha256_bytes(config_path.read_bytes())[:8]}"


def build_silver(dataset_version_id: str, config: dict) -> tuple[pd.DataFrame, dict]:
    """Rename, cast and role-tag Bronze rows using the YAML contract only."""
    df, roles = load_and_canonicalize(dataset_version_id, config)
    return df, roles


def check_gates(df: pd.DataFrame, roles: dict, config: dict) -> list[dict]:
    """Hard gates. Any failure blocks the Parquet write."""
    contract = config["silver_eda_contract"]
    names = list(contract["canonical_names"].values())
    dtypes = contract["canonical_dtypes"]
    expected_rows = int(config["bronze_source_contract"]["rows"])

    results: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    add("row_count", len(df) == expected_rows, f"observed={len(df)} expected={expected_rows}")

    missing = [c for c in names if c not in df.columns]
    add("expected_columns", not missing, f"missing={missing}")

    present = [c for c in names if c in df.columns]
    policy = contract.get("missingness", {})
    default_limit = float(policy.get("default_max_rate", 0.0))
    limits = policy.get("max_rate_by_column", {})

    for col in present:
        limit = float(limits.get(col, default_limit))
        if not 0.0 <= limit <= 1.0:
            raise ValueError(f"{col}: missing-rate limit must be in [0, 1]")

        missing_rate = float(df[col].isna().mean())
        add(
            f"missing_rate:{col}",
            missing_rate <= limit,
            f"observed={missing_rate:.6f}, maximum={limit:.6f}",
        )

    bad_dtypes = {
        c: {"expected": dtypes[c], "observed": str(df[c].dtype)}
        for c in present
        if c in dtypes and dtypes[c] not in TEXT_DTYPES and str(df[c].dtype) != dtypes[c]
    }
    add("declared_dtypes", not bad_dtypes, f"mismatches={bad_dtypes}")

    task = config.get("data", {}).get("task", "binary_classification")

    if task == "binary_classification":
        bad_binary = {
            c: sorted(set(df[c].dropna().unique()) - {0, 1})
            for c in roles["target"]
            if c in df.columns
        }
        bad_binary = {c: v for c, v in bad_binary.items() if v}
        add(
            "binary_fields_in_0_1",
            not bad_binary,
            f"violations={bad_binary}",
        )
    elif task == "regression":
        for col in roles["target"]:
            if col in df.columns:
                add(
                    f"target_numeric:{col}",
                    pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_bool_dtype(df[col]),
                    f"dtype={df[col].dtype}",
                )
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    allowed_values = contract.get("allowed_values", {})

    for col, allowed_categories in allowed_values.items():
        if col not in df.columns:
            add(
                f"allowed_values:{col}",
                False,
                "configured column is missing",
            )
            continue

        values = df[col].dropna()
        unexpected = values.loc[~values.isin(allowed_categories)].unique().tolist()

        add(
            f"allowed_values:{col}",
            not unexpected,
            f"unexpected={unexpected}",
        )

    raw_key = config.get("data", {}).get("entity_id")
    key = contract["canonical_names"].get(raw_key, raw_key) if raw_key else None
    if key is None:
        key = next(iter(roles.get("id", [])), None)
    add("entity_key_configured", key is not None, f"key={key!r}")
    if key is not None:
        exists = key in df.columns
        add("entity_key_exists", exists, f"key={key!r}")
        add("entity_key_in_id_role", key in roles.get("id", []), f"key={key!r}")
        if exists:
            nulls = int(df[key].isna().sum())
            add("entity_key_non_null", nulls == 0, f"key={key!r}, nulls={nulls}")
            if config.get("validation", {}).get("require_unique_entity_id", True):
                duplicates = int(df[key].duplicated().sum())
                add("id_unique", duplicates == 0, f"key={key!r}, duplicates={duplicates}")
    for target_col in roles.get("target", []):
        exists = target_col in df.columns
        add(f"target_exists:{target_col}", exists, f"column={target_col!r}")
        if exists:
            nulls = int(df[target_col].isna().sum())
            add(f"target_non_null:{target_col}", nulls == 0, f"nulls={nulls}")

    source_cols = [c for c in df.columns if c not in {"dataset_version_id", "source_row_number"}]
    dup_rows = int(df.duplicated(subset=source_cols).sum())
    add("no_duplicate_source_records", dup_rows == 0, f"duplicates={dup_rows}")

    return results


def write_outputs(
    df: pd.DataFrame,
    roles: dict,
    gates: list[dict],
    *,
    dataset_name: str,
    dataset_version_id: str,
    config_path: Path,
    silver_root: Path,
    artifact_root: Path,
) -> int:
    config_hash = sha256_bytes(config_path.read_bytes())
    silver_version_id = get_silver_version_id(dataset_version_id, config_path)
    passed = all(g["passed"] for g in gates)

    quality_dir = artifact_root / dataset_name / silver_version_id
    quality_dir.mkdir(parents=True, exist_ok=True)

    validation = {
        "silver_dataset_version_id": silver_version_id,
        "dataset_version_id": dataset_version_id,
        "validated_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "gates": gates,
    }
    (quality_dir / "validation.json").write_text(json.dumps(validation, indent=2))

    if not passed:
        logger.error("Silver build FAILED validation; no Parquet written.")
        for g in gates:
            if not g["passed"]:
                logger.error("  %s: %s", g["check"], g["detail"])
        return 1

    out_dir = silver_root / dataset_name / silver_version_id
    if out_dir.exists():
        raise FileExistsError(f"Silver version already exists (immutable): {out_dir}")
    out_dir.mkdir(parents=True)
    parquet_path = out_dir / "data.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    manifest = {
        "silver_dataset_version_id": silver_version_id,
        "dataset_version_id": dataset_version_id,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "roles": {k: list(v) for k, v in roles.items()},
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "parquet_path": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "validation_report": str(quality_dir / "validation.json"),
        "validation_passed": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s (%d rows)", parquet_path, len(df))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Silver Parquet from Bronze.")
    p.add_argument("--dataset-version-id", required=True)
    p.add_argument("--config", default="configs/ai4i_binary.yaml")
    p.add_argument("--dataset-name", default="ai4i")
    p.add_argument("--silver-root", default="data/silver")
    p.add_argument("--artifact-root", default="artifacts/quality")
    args = p.parse_args(argv)

    setup_logging()
    logger.info("Start silver process")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    df, roles = build_silver(args.dataset_version_id, config)
    gates = check_gates(df, roles, config)

    exit_code = write_outputs(
        df,
        roles,
        gates,
        dataset_name=args.dataset_name,
        dataset_version_id=args.dataset_version_id,
        config_path=config_path,
        silver_root=Path(args.silver_root),
        artifact_root=Path(args.artifact_root),
    )
    version = get_silver_version_id(args.dataset_version_id, config_path)
    out_dir = Path(args.silver_root) / args.dataset_name / version
    validation_path = Path(args.artifact_root) / args.dataset_name / version / "validation.json"
    manifest = {}
    if exit_code == 0:
        manifest = json.loads((out_dir / "manifest.json").read_text())
    data_run_registry.register_silver_build(
        get_engine(),
        silver_dataset_version_id=version,
        dataset_version_id=args.dataset_version_id,
        succeeded=exit_code == 0,
        validation_path=str(validation_path),
        manifest_path=str(out_dir / "manifest.json") if exit_code == 0 else None,
        parquet_path=manifest.get("parquet_path"),
        row_count=manifest.get("row_count"),
        config_sha256=manifest.get("config_sha256"),
        parquet_sha256=manifest.get("parquet_sha256"),
        error_summary=None if exit_code == 0 else "validation gates failed",
    )

    logger.info("Silver dataset done")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
