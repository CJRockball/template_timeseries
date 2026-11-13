"""Read the dataset contract (e.g. configs/ai4i_binary.yaml) in canonical names.

The template's standard dataset is a key, features and one binary target. Other
role lists in the YAML (such as ``outcome``) are ignored here. Roles live under
``bronze_source_contract`` (as ingest reads them) or at the top level.

Training code asks this module which column is the key, the target or a feature.
Nothing downstream hardcodes dataset column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetContract:
    entity_id: str
    target: str
    ids: tuple[str, ...]
    features: tuple[str, ...]
    dtypes: dict[str, str] = field(default_factory=dict)  # canonical feature dtypes

    @property
    def excluded_from_features(self) -> tuple[str, ...]:
        """Columns that must never reach a model: ids and the target."""
        return (*self.ids, self.target)


def _find_roles(raw: dict[str, Any]) -> dict[str, Any]:
    for section in (raw.get("bronze_source_contract") or {}, raw):
        if isinstance(section, dict) and isinstance(section.get("roles"), dict):
            return section["roles"]
    raise KeyError("roles")


def _role(roles: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        if key in roles:
            return list(roles[key] or [])
    return []


def load_dataset_contract(path: str | Path) -> DatasetContract:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        names: dict[str, str] = raw["silver_eda_contract"]["canonical_names"]
        dtypes: dict[str, str] = raw["silver_eda_contract"].get("canonical_dtypes", {})
        roles: dict[str, Any] = _find_roles(raw)
        entity_raw: str = raw["data"]["entity_id"]
        target_raw: str = raw["data"]["target"]
    except KeyError as exc:
        raise ValueError(f"{path}: missing section or key {exc}") from exc

    def canon(columns: list[str]) -> tuple[str, ...]:
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValueError(f"{path}: no canonical name for {missing}")
        return tuple(names[c] for c in columns)

    ids = canon(_role(roles, "id", "ids"))
    features = canon(_role(roles, "feature", "features"))
    targets = canon(_role(roles, "target", "targets"))
    if len(targets) != 1:
        raise ValueError(f"{path}: single-target template, got target roles {targets}")
    target = targets[0]
    if canon([target_raw])[0] != target:
        raise ValueError(f"{path}: data.target disagrees with roles.target")
    entity_id = canon([entity_raw])[0]
    if entity_id not in ids:
        raise ValueError(f"{path}: entity_id {entity_id!r} must have the id role")

    seen: dict[str, str] = {}
    for group, cols in {"ids": ids, "features": features, "target": (target,)}.items():
        for col in cols:
            if col in seen:
                raise ValueError(f"{path}: {col!r} is in both {seen[col]} and {group}")
            seen[col] = group

    return DatasetContract(
        entity_id=entity_id,
        target=target,
        ids=ids,
        features=features,
        dtypes={c: dtypes[c] for c in features if c in dtypes},
    )
