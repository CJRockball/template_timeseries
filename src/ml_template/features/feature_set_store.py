"""Store feature blocks and feature sets, and resolve a set into a model frame.

Layout under ``data/features/``::

    <block_id>/data.parquet      new columns only, keyed by the entity key
    <block_id>/manifest.json     lineage: Silver ID, code hash, params, columns, hashes
    _sets/<feature_set_id>.json  one feature set: parent, blocks, final column list

Files are immutable and content-addressed. A block ID covers the Silver version, the block
code and the parameters. A feature set ID covers the Silver version, the ordered blocks, the
final columns and the parent. Database rows only index these files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml_template.features.block_registry import get_block
from ml_template.paths import DATA_FEATURES
from ml_template.tracking.run_artifacts import sha256_file

logger = logging.getLogger("ml_template.features.feature_set_store")

SETS_DIR = "_sets"
SILVER_SOURCE = "silver"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- blocks ------------------------------------------------------------------


def build_block(
    name: str,
    features: pd.DataFrame,
    silver_id: str,
    params: Mapping[str, Any] | None = None,
    root: Path = DATA_FEATURES,
) -> dict[str, Any]:
    """Compute one block from the Silver feature columns and store it. Returns its manifest.

    ``features`` holds only feature-role columns, indexed by the entity key. An existing
    block with the same ID is verified against its hash and reused.
    """
    key_col = features.index.name
    if not key_col:
        raise ValueError("features must be indexed by the named entity key")
    block = get_block(name)
    resolved = block.resolve_params(params)
    block_id = block.block_id(silver_id, resolved)
    out_dir = Path(root) / block_id
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        load_block(block_id, root)  # verifies the Parquet hash
        logger.info("Block %s already exists and matches; reusing", block_id)
        return manifest

    out = block.run(features, resolved)
    observed: dict[str, float] = {}
    if block.sanity is not None:
        observed = {k: float(v) for k, v in block.sanity(features, out, resolved).items()}
    sanity = {}
    for key, value in observed.items():
        expected = block.expected.get(key)
        ok = expected is None or value == expected
        sanity[key] = {"observed": value, "expected": expected, "ok": ok}
        if not ok:
            logger.warning("Block %s sanity %s = %s, expected %s", block_id, key, value, expected)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "data.parquet"
    out.reset_index().to_parquet(parquet_path, index=False)
    manifest = {
        "block_id": block_id,
        "block_name": name,
        "silver_dataset_version_id": silver_id,
        "key_col": key_col,
        "code_sha256": block.code_sha256(),
        "params": resolved,
        "derived_from": list(block.required_columns(resolved)),
        "description": block.description,
        "columns": list(out.columns),
        "dtypes": {c: str(t) for c, t in out.dtypes.items()},
        "n_rows": int(len(out)),
        "parquet_path": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "sanity": sanity,
        "created_at": _now(),
    }
    _write_json(manifest_path, manifest)
    logger.info("Wrote block %s (%d rows, columns %s)", block_id, len(out), list(out.columns))
    return manifest


def load_block(block_id: str, root: Path = DATA_FEATURES) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the block manifest and its columns indexed by the entity key."""
    out_dir = Path(root) / block_id
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"block {block_id} not found at {out_dir}; build it first")
    manifest = _read_json(manifest_path)
    parquet_path = out_dir / "data.parquet"
    if sha256_file(parquet_path) != manifest["parquet_sha256"]:
        raise ValueError(f"{parquet_path} differs from the hash in its manifest")
    frame = pd.read_parquet(parquet_path).set_index(manifest["key_col"])
    return manifest, frame


# --- feature sets ------------------------------------------------------------


def _set_path(feature_set_id: str, root: Path) -> Path:
    return Path(root) / SETS_DIR / f"{feature_set_id}.json"


def load_feature_set(feature_set_id: str, root: Path = DATA_FEATURES) -> dict[str, Any]:
    path = _set_path(feature_set_id, root)
    if not path.exists():
        raise FileNotFoundError(f"feature set {feature_set_id} not found at {path}")
    return _read_json(path)


def create_feature_set(
    silver_id: str,
    block_ids: Sequence[str],
    *,
    raw_columns: Sequence[str],
    columns: Sequence[str] | None = None,
    parent: str | None = None,
    description: str = "",
    root: Path = DATA_FEATURES,
) -> dict[str, Any]:
    """Register a feature set: Silver feature columns plus the columns of some blocks.

    ``columns`` optionally selects the final model columns (raw or block columns), in the
    order given. A selection is a new feature set; it never creates a new data file.
    """
    manifests = []
    for block_id in block_ids:
        manifest = _read_json(Path(root) / block_id / "manifest.json")
        if manifest["silver_dataset_version_id"] != silver_id:
            raise ValueError(f"block {block_id} was built from another Silver version")
        manifests.append(manifest)

    sources = {c: SILVER_SOURCE for c in raw_columns}
    for manifest in manifests:
        for col in manifest["columns"]:
            if col in sources:
                raise ValueError(f"duplicate column {col!r} in feature set")
            sources[col] = manifest["block_name"]

    if columns is None:
        final = list(sources)
    else:
        final = list(columns)
        unknown = sorted(set(final) - set(sources))
        if unknown or len(set(final)) != len(final) or not final:
            raise ValueError(f"invalid column selection (unknown: {unknown})")

    if parent is not None:
        parent_manifest = load_feature_set(parent, root)
        if parent_manifest["silver_dataset_version_id"] != silver_id:
            raise ValueError(f"parent {parent} was built from another Silver version")

    identity = {
        "silver": silver_id,
        "blocks": list(block_ids),
        "columns": final,
        "parent": parent,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    feature_set_id = f"fs_{digest[:12]}"
    path = _set_path(feature_set_id, root)
    if path.exists():
        logger.info("Feature set %s already exists; reusing", feature_set_id)
        return _read_json(path)

    manifest = {
        "feature_set_id": feature_set_id,
        "silver_dataset_version_id": silver_id,
        "parent_feature_set_id": parent,
        "blocks": list(block_ids),
        "raw_columns": list(raw_columns),
        "selection": None if columns is None else final,
        "columns": final,
        "column_sources": {c: sources[c] for c in final},
        "n_features": len(final),
        "description": description,
        "created_at": _now(),
    }
    _write_json(path, manifest)
    logger.info("Wrote feature set %s with %d features", feature_set_id, len(final))
    return manifest


def extend_feature_set(
    parent: str,
    new_block_ids: Sequence[str],
    *,
    description: str = "",
    root: Path = DATA_FEATURES,
) -> dict[str, Any]:
    """New feature set = parent plus new blocks, with ``parent_feature_set_id`` set."""
    base = load_feature_set(parent, root)
    added = [b for b in new_block_ids if b not in base["blocks"]]
    if not added:
        return base
    columns = None
    if base["selection"] is not None:
        columns = list(base["selection"])
        for block_id in added:
            columns += _read_json(Path(root) / block_id / "manifest.json")["columns"]
    return create_feature_set(
        base["silver_dataset_version_id"],
        [*base["blocks"], *added],
        raw_columns=base["raw_columns"],
        columns=columns,
        parent=parent,
        description=description,
        root=root,
    )


def resolve_feature_set(
    feature_set_id: str,
    raw_features: pd.DataFrame,
    silver_id: str,
    root: Path = DATA_FEATURES,
) -> pd.DataFrame:
    """Join a feature set's block columns onto the Silver features.

    Guards: same Silver version, same Silver feature columns, and every key present in both
    sides. Row order follows ``raw_features``.
    """
    manifest = load_feature_set(feature_set_id, root)
    if manifest["silver_dataset_version_id"] != silver_id:
        raise ValueError(
            f"feature set {feature_set_id} belongs to {manifest['silver_dataset_version_id']}, "
            f"not {silver_id}"
        )
    if set(manifest["raw_columns"]) != set(raw_features.columns):
        raise ValueError(f"Silver feature columns changed since {feature_set_id} was created")

    parts = [raw_features]
    for block_id in manifest["blocks"]:
        _, block_frame = load_block(block_id, root)
        if set(block_frame.index) != set(raw_features.index):
            raise ValueError(f"keys of block {block_id} do not match the Silver rows")
        parts.append(block_frame.reindex(raw_features.index))
    joined = pd.concat(parts, axis=1)
    if joined.columns.duplicated().any():
        raise ValueError(f"duplicate columns after joining {feature_set_id}")
    return joined[manifest["columns"]]


# --- database index ----------------------------------------------------------


def register_block_record(engine: Any, manifest: Mapping[str, Any]) -> None:
    """Insert a row in ``feature_blocks`` (created if missing). Idempotent."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from ml_template.db.tracking_tables import ensure_tracking_tables, feature_blocks

    ensure_tracking_tables(engine)
    stmt = (
        sqlite_insert(feature_blocks)
        .values(
            block_id=manifest["block_id"],
            block_name=manifest["block_name"],
            silver_dataset_version_id=manifest["silver_dataset_version_id"],
            code_sha256=manifest["code_sha256"],
            params_json=json.dumps(manifest["params"], sort_keys=True),
            columns_json=json.dumps(manifest["columns"]),
            derived_from_json=json.dumps(manifest["derived_from"]),
            description=manifest["description"],
            n_rows=manifest["n_rows"],
            parquet_path=manifest["parquet_path"],
            parquet_sha256=manifest["parquet_sha256"],
            sanity_json=json.dumps(manifest["sanity"], sort_keys=True),
            created_at=manifest["created_at"],
        )
        .on_conflict_do_nothing(index_elements=["block_id"])
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def register_feature_set_record(
    engine: Any, manifest: Mapping[str, Any], root: Path = DATA_FEATURES
) -> None:
    """Insert a row in ``feature_sets`` (created if missing). Idempotent."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from ml_template.db.tracking_tables import ensure_tracking_tables, feature_sets

    ensure_tracking_tables(engine)
    stmt = (
        sqlite_insert(feature_sets)
        .values(
            feature_set_id=manifest["feature_set_id"],
            silver_dataset_version_id=manifest["silver_dataset_version_id"],
            parent_feature_set_id=manifest["parent_feature_set_id"],
            blocks_json=json.dumps(manifest["blocks"]),
            columns_json=json.dumps(manifest["columns"]),
            n_features=manifest["n_features"],
            description=manifest["description"],
            manifest_path=str(_set_path(manifest["feature_set_id"], root)),
            created_at=manifest["created_at"],
        )
        .on_conflict_do_nothing(index_elements=["feature_set_id"])
    )
    with engine.begin() as conn:
        conn.execute(stmt)
