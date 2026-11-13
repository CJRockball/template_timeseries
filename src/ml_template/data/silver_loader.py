"""Load a Silver dataset for modelling: key, features and one binary target.

Roles come from the Silver manifest, which records the roles used when that exact
version was built. Only the ``id``, ``feature`` and ``target`` roles are used; any
other role (for example ``outcome``) is ignored, and only the columns the model
needs are read from the Parquet file. Features are chosen by explicit inclusion,
so lineage columns such as ``dataset_version_id`` can never reach a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_template.paths import DATA_SILVER

REQUIRED_ROLES = ("id", "feature", "target")
ROLE_ALIASES = {"ids": "id", "features": "feature", "targets": "target"}


@dataclass(frozen=True)
class SilverData:
    silver_dataset_version_id: str
    key_col: str
    df: pd.DataFrame  # key, feature and target columns only
    roles: dict[str, list[str]]  # normalised: id, feature, target
    manifest: dict[str, Any]
    parquet_path: Path

    @property
    def features(self) -> list[str]:
        return list(self.roles["feature"])

    @property
    def target(self) -> str:
        return self.roles["target"][0]

    def model_frame(self) -> pd.DataFrame:
        """Rows indexed by the entity key and sorted by it: features plus the target."""
        cols = [self.key_col, *self.features, self.target]
        return self.df[cols].set_index(self.key_col).sort_index()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_roles(raw_roles: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for name, columns in raw_roles.items():
        canonical = ROLE_ALIASES.get(name, name)
        if canonical in REQUIRED_ROLES:
            roles.setdefault(canonical, []).extend(list(columns or []))
    return roles


def load_silver(
    silver_dataset_version_id: str,
    dataset_name: str,
    key_col: str,
    silver_root: Path = DATA_SILVER,
    verify_hash: bool = True,
    *,
    task: str = "binary_classification",
) -> SilverData:
    out_dir = Path(silver_root) / dataset_name / silver_dataset_version_id
    manifest_path = out_dir / "manifest.json"
    parquet_path = out_dir / "data.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; build Silver first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("silver_dataset_version_id") != silver_dataset_version_id:
        raise ValueError(f"{manifest_path} belongs to a different Silver version")
    if not manifest.get("validation_passed"):
        raise ValueError(f"{silver_dataset_version_id} did not pass validation")
    if verify_hash:
        observed = _sha256_file(parquet_path)
        if observed != manifest["parquet_sha256"]:
            raise ValueError(f"{parquet_path} differs from the hash in its manifest")

    roles = _normalise_roles(manifest["roles"])
    missing = [r for r in REQUIRED_ROLES if not roles.get(r)]
    if missing:
        raise ValueError(f"manifest has no columns for roles {missing}")
    if len(roles["target"]) != 1:
        raise ValueError(
            f"single-target template: manifest lists {len(roles['target'])} target columns "
            f"{roles['target']}"
        )
    if key_col not in roles["id"]:
        raise ValueError(f"key column {key_col!r} does not have the id role")
    overlap = sorted(set(roles["feature"]) & (set(roles["id"]) | set(roles["target"])))
    if overlap:
        raise ValueError(f"feature role overlaps id or target columns: {overlap}")

    needed = [key_col, *roles["feature"], roles["target"][0]]
    absent = [c for c in needed if c not in manifest["columns"]]
    if absent:
        raise ValueError(f"{silver_dataset_version_id} manifest has no columns {absent}")

    df = pd.read_parquet(parquet_path, columns=needed)
    if df[key_col].isna().any() or df[key_col].duplicated().any():
        raise ValueError(f"{key_col} must be unique and non-null")
    target = roles["target"][0]
    values = df[target]

    if task == "binary_classification":
        if not values.isin([0, 1]).all():
            raise ValueError(f"target {target!r} must be binary 0/1")

    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"target {target!r} must be numeric")

        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all():
            raise ValueError(f"target {target!r} must be finite with no nulls")

    else:
        raise ValueError(f"unsupported task: {task!r}")

    return SilverData(
        silver_dataset_version_id=silver_dataset_version_id,
        key_col=key_col,
        df=df,
        roles=roles,
        manifest=manifest,
        parquet_path=parquet_path,
    )
