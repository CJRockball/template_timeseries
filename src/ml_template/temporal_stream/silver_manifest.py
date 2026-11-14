"Manifest identities and inventory verification, usable without Arrow."

import json
from pathlib import Path, PurePosixPath

from .silver_schema import FORMAT, TemporalSchema, file_hash, sha


def seal(manifest):
    value = dict(manifest)
    value.pop("manifest_sha256", None)
    return {**value, "manifest_sha256": sha(value)}


def load_manifest(root):
    root = Path(root).resolve()
    if (root / "manifest.json").is_symlink():
        raise ValueError("Symlink manifest rejected")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != seal(manifest)["manifest_sha256"]:
        raise ValueError("Manifest integrity failure")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("dataset_kind") != "temporal"
        or manifest.get("eligible_for_legacy_training") is not False
    ):
        raise ValueError("Not a supported temporal Silver dataset")
    schema = TemporalSchema.from_dict(manifest["temporal_schema"])
    if manifest.get("roles") != schema.roles or manifest.get("order_by") != schema.order:
        raise ValueError("Temporal roles/order mismatch")
    if manifest["identity"]["temporal_schema"] != schema.spec:
        raise ValueError("Schema is not bound to identity")
    if manifest["identity"]["input"] != manifest["input"]:
        raise ValueError("Input lineage mismatch")
    if manifest["identity"]["logical_data_sha256"] != manifest["logical_data_sha256"]:
        raise ValueError("Logical hash identity mismatch")
    if manifest["silver_dataset_version_id"] != "silver_ts_" + sha(manifest["identity"]):
        raise ValueError("Silver identity mismatch")
    if type(manifest["row_count"]) is not int or manifest["row_count"] < 1:
        raise ValueError("Invalid row count")
    if not manifest.get("validation_passed") or not manifest.get("files"):
        raise ValueError("Unvalidated/empty Silver")
    return manifest


def verify_inventory(root):
    root = Path(root).resolve()
    manifest = load_manifest(root)
    listed = set()
    total = 0
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("data",)
            or (relative.suffix != ".parquet")
        ):
            raise ValueError("Unsafe Parquet inventory path")
        if item["path"] in listed or type(item["rows"]) is not int or item["rows"] < 1:
            raise ValueError("Duplicate file or invalid file count")
        listed.add(item["path"])
        file = root.joinpath(*relative.parts)
        if file.is_symlink() or not file.is_file() or (not file.resolve().is_relative_to(root)):
            raise ValueError("Missing/unsafe Parquet file")
        if any(p.is_symlink() for p in file.parents if p != root.parent):
            raise ValueError("Symlink path rejected")
        if file.stat().st_size != item["bytes"] or file_hash(file) != item["sha256"]:
            raise ValueError("Parquet integrity failure")
        total += item["rows"]
    actual = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
    if listed != actual or total != manifest["row_count"]:
        raise ValueError("Parquet inventory/count mismatch")
    report = root / "validation.json"
    if report.is_symlink() or file_hash(report) != manifest["validation_sha256"]:
        raise ValueError("Validation report integrity failure")
    validation = json.loads(report.read_text(encoding="utf-8"))
    if validation.get("passed") is not True or validation.get("rows") != manifest["row_count"]:
        raise ValueError("Validation report failed")
    return manifest
