"""Hashed monitoring artifacts. Integrity checks are not authentication."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np


def now():
    return datetime.now(UTC).isoformat()


def new_id(prefix):
    return f"{prefix}_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid4().hex[:12]}"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def safe_path(root, name):
    part = PurePosixPath(name)
    if (
        not name
        or part.is_absolute()
        or ".." in part.parts
        or "\\" in name
        or part.as_posix() != name
    ):
        raise ValueError("unsafe artifact name")
    root = Path(root).resolve()
    current = root
    for segment in part.parts:
        current = current / segment
        if current.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
    if not current.resolve().is_relative_to(root):
        raise ValueError("artifact escapes root")
    return current


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
        name = path.relative_to(root).as_posix()
        if path.is_file() and name != "artifacts.json":
            result[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def seal(root):
    root = Path(root)
    write_json(root / "artifacts.json", {"files": inventory(root)})
    return digest(root / "artifacts.json")


def verify(root, expected_manifest_sha256=None):
    root = Path(root).resolve()
    manifest = safe_path(root, "artifacts.json")
    if expected_manifest_sha256 is not None and digest(manifest) != expected_manifest_sha256:
        raise ValueError("artifact manifest differs from registry")
    value = read_json(manifest)
    if set(value) != {"files"} or not isinstance(value["files"], dict):
        raise ValueError("invalid artifact manifest")
    for name in value["files"]:
        safe_path(root, name)
    if inventory(root) != value["files"]:
        raise ValueError("artifact hash, size or file inventory mismatch")
    return value["files"]
