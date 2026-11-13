"""Strict content manifests for explanation outputs and scoring inputs."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(root, name):
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if not name or part.is_absolute() or ".." in part.parts or "\\" in name:
        raise ValueError("unsafe artifact path")
    if part.as_posix() != name:
        raise ValueError("noncanonical artifact path")
    path = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError("artifact symlinks are unsupported")
    if not path.resolve().is_relative_to(root):
        raise ValueError("artifact path escapes root")
    return path


def file_entries(root, *, exclude=("artifacts.json",)):
    root = Path(root).resolve()
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are unsupported")
        name = path.relative_to(root).as_posix()
        if path.is_file() and name not in exclude:
            out[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return out


def verify_files(root, entries, *, exclude=("artifacts.json",)):
    if not isinstance(entries, dict) or not entries:
        raise ValueError("nonempty artifact file manifest required")
    for name in entries:
        safe_path(root, name)
    if file_entries(root, exclude=exclude) != entries:
        raise ValueError("artifact missing, changed or unlisted")


def seal_explanation(root, explanation_id):
    payload = {
        "schema_version": 1,
        "kind": "model_explanation",
        "explanation_id": explanation_id,
        "files": file_entries(root),
    }
    manifest = {**payload, "content_sha256": fingerprint(payload)}
    write_json(Path(root) / "artifacts.json", manifest)
    return manifest


def verify_explanation(root, expected_id=None):
    manifest = read_json(safe_path(root, "artifacts.json"))
    if set(manifest) != {"schema_version", "kind", "explanation_id", "files", "content_sha256"}:
        raise ValueError("unsupported explanation manifest fields")
    if manifest["schema_version"] != 1 or manifest["kind"] != "model_explanation":
        raise ValueError("unsupported explanation manifest")
    if expected_id is not None and manifest["explanation_id"] != expected_id:
        raise ValueError("unexpected explanation identity")
    payload = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if fingerprint(payload) != manifest["content_sha256"]:
        raise ValueError("explanation content identity mismatch")
    required = {
        "config.json",
        "summary.json",
        "provenance.json",
        "rows.parquet",
        "features.parquet",
        "contributions.parquet",
        "global_summary.parquet",
        "feature_metadata.json",
        "verification.json",
        "report.md",
        "report.html",
    }
    if not required <= manifest["files"].keys():
        raise ValueError("incomplete explanation package")
    verify_files(root, manifest["files"])
    return manifest
