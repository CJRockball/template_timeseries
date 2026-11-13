"""Resolve a run config into model inputs: features, target and stored split.

Shared by training and holdout evaluation so both see identical data. The model
sees only the key, the feature columns and the single target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_template.data.contract import DatasetContract, load_dataset_contract
from ml_template.data.silver_loader import SilverData, load_silver
from ml_template.features.feature_set_store import resolve_feature_set
from ml_template.models.model_factory import prepare_features
from ml_template.paths import DATA_FEATURES, DATA_SILVER, DATA_SPLITS
from ml_template.training.run_config import RunConfig, resolve_path
from ml_template.training.split import compute_split_id, load_split


@dataclass(frozen=True)
class RunInputs:
    contract: DatasetContract
    silver: SilverData
    X: pd.DataFrame  # feature columns only, indexed by entity key
    y: pd.Series  # binary target, same index
    assignments: pd.DataFrame  # key, partition, fold
    split_id: str


def split_id_for(cfg: RunConfig, contract: DatasetContract) -> str:
    return compute_split_id(
        cfg.data.silver_dataset_version_id, cfg.split, contract.entity_id, contract.target
    )


def load_run_inputs(
    cfg: RunConfig,
    *,
    silver_root: Path = DATA_SILVER,
    split_root: Path = DATA_SPLITS,
    feature_root: Path = DATA_FEATURES,
) -> RunInputs:
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    silver = load_silver(
        cfg.data.silver_dataset_version_id,
        cfg.data.dataset_name,
        contract.entity_id,
        silver_root,
        task=cfg.task,
    )
    if silver.target != contract.target or set(silver.features) != set(contract.features):
        raise ValueError(
            "Silver manifest roles and the dataset config disagree; "
            "rebuild Silver or restore the config used for it"
        )

    split_id = split_id_for(cfg, contract)
    assignments = load_split(split_id, split_root)
    if cfg.task == "regression" and (
        cfg.split.calibration_fraction or cfg.split.recalibration_fraction
    ):
        import hashlib
        import json

        from ml_template.training.regression_partitions import validate_assignments

        manifest = json.loads((Path(split_root) / split_id / "manifest.json").read_text())
        validate_assignments(cfg.split, assignments, manifest)
        dev = assignments.partition.eq("dev")
        key = contract.entity_id
        if (
            assignments[key].isna().any()
            or assignments[key].duplicated().any()
            or len(assignments) != manifest["n_rows"]
            or int(assignments.partition.eq("holdout").sum()) != manifest["n_holdout"]
            or not assignments.loc[~dev, "fold"].eq(-1).all()
            or sorted(assignments.loc[dev, "fold"].unique()) != list(range(cfg.split.n_folds))
            or hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
            != manifest["assignments_sha256"]
        ):
            raise ValueError("invalid reserved split inventory before fitting")

    frame = silver.model_frame()
    raw = frame[silver.features]
    if cfg.data.feature_set_id is not None:
        raw = resolve_feature_set(
            cfg.data.feature_set_id, raw, silver.silver_dataset_version_id, feature_root
        )
    X = prepare_features(raw)
    import numpy as np
    import yaml

    raw_contract = yaml.safe_load(resolve_path(cfg.data.dataset_config).read_text(encoding="utf-8"))
    contract_task = raw_contract.get("data", {}).get("task", "binary_classification")
    if contract_task != cfg.task:
        raise ValueError("Experiment task disagrees with dataset contract")
    target = frame[silver.target]
    if cfg.task == "regression":
        if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
            raise ValueError("Regression target must be numeric")
        y = target.astype(float)
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError("Regression target must be finite")
    else:
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError("Binary target must be 0/1 without nulls")
        y = target.astype(int)
    if set(assignments[contract.entity_id]) != set(frame.index):
        raise ValueError(
            f"split {split_id} does not cover the rows of {silver.silver_dataset_version_id}"
        )
    return RunInputs(contract, silver, X, y, assignments, split_id)
