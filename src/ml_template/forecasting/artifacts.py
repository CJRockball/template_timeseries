"Immutable inventory verification and bounded JSONL reads. No Arrow dependency."

import hashlib
import json
from pathlib import Path

from .policy import FORMAT, canonical, digest

FILES = ("origins.jsonl", "eligibility.jsonl", "requests.jsonl", "validation.json")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def seal(value):
    value = dict(value)
    value.pop("manifest_sha256", None)
    return {**value, "manifest_sha256": digest(value)}


def verify_plan(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symlink plan rejected")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Symlink manifest rejected")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("manifest_sha256") != seal(value)["manifest_sha256"]
        or value.get("format") != FORMAT
    ):
        raise ValueError("Plan manifest integrity failure")
    if value["plan_id"] != "origin_" + digest(value["identity"]):
        raise ValueError("Plan identity failure")
    if set(value["files"]) != set(FILES):
        raise ValueError("Plan inventory mismatch")
    for name in FILES:
        file = path / name
        info = value["files"][name]
        if (
            file.is_symlink()
            or not file.is_file()
            or file.stat().st_size != info["bytes"]
            or (file_hash(file) != info["sha256"])
        ):
            raise ValueError("Plan file integrity failure: " + name)
    report = json.loads((path / "validation.json").read_text(encoding="utf-8"))
    if report.get("passed") is not True or report["requests"] < 1:
        raise ValueError("Unvalidated/empty plan")
    if report != value["validation"]:
        raise ValueError("Plan validation mismatch")
    return value


def iter_plan_rows(path, artifact="requests"):
    if artifact not in {"requests", "origins", "eligibility"}:
        raise ValueError("Unknown plan artifact")
    verify_plan(path)
    with (Path(path) / (artifact + ".jsonl")).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_line(handle, value):
    handle.write(canonical(value) + "\n")
