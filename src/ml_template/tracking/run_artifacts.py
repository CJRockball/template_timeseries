"""Run folder layout and integrity records.

One folder per run under artifacts/runs/<run_id>/. The database indexes runs and
metrics; this folder holds the files. ``artifact_manifest.json`` lists every file
with its size and SHA-256, so a stored model can be verified before it is used.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ml_template.paths import ARTIFACTS_RUNS, ROOT

MANIFEST_NAME = "artifact_manifest.json"
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "torch",
    "pyarrow",
    "pydantic",
    "sqlalchemy",
)


def create_run_dir(run_id: str, root: Path = ARTIFACTS_RUNS) -> Path:
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def write_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """(Re)write the manifest for every file currently in the run folder."""
    run_dir = Path(run_dir)
    files = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME):
        files[path.relative_to(run_dir).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {"files": files}
    write_json(run_dir / MANIFEST_NAME, manifest)
    return manifest


def verify_artifact(run_dir: Path, relative_name: str) -> None:
    """Raise if a file differs from the manifest recorded at the end of the run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; the run did not finish cleanly")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["files"].get(relative_name)
    if recorded is None:
        raise KeyError(f"{relative_name} is not listed in {manifest_path}")
    if sha256_file(run_dir / relative_name) != recorded["sha256"]:
        raise ValueError(f"{run_dir / relative_name} differs from its recorded hash")


def relative_to_root(path: Path) -> str:
    """Store paths relative to the repo when possible, so a moved checkout still works."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_stored_path(stored: str) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else ROOT / path
