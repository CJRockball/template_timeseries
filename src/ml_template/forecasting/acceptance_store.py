"""Independent report/resource artifacts; never changes learned producer identities."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path, PurePosixPath

FORMAT = "temporal-acceptance-v1"
REQUIRED = {
    "report": {"report.json", "summary.md", "metrics.csv"},
    "resource": {"resource.json", "stdout.log"},
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path, name):
    with (Path(path) / name).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def inventory(root, expected):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Symlink artifact rejected")
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if names != {"manifest.json", *expected}:
        raise ValueError("Artifact inventory mismatch")
    for name, info in expected.items():
        parts = PurePosixPath(name).parts
        if (
            PurePosixPath(name).is_absolute()
            or chr(92) in name
            or ".." in parts
            or not parts
            or (len(parts) > 1 and (len(parts) != 2 or parts[0] != "models"))
        ):
            raise ValueError("Unsafe artifact inventory path")
        path = root / name
        if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
            raise ValueError("Artifact file integrity failure")


def input_artifact(root, kind):
    root = Path(root)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("Symlink artifact rejected")
    value = load(root / "manifest.json")
    if digest({k: v for k, v in value.items() if k != "manifest_sha256"}) != value.get(
        "manifest_sha256"
    ):
        raise ValueError("Manifest integrity failure")
    identity = value["identity"]
    if identity.get("format") != value["format"] or identity["kind"] != kind:
        raise ValueError("Wrong input artifact kind")
    history = value["format"] == "temporal-history-v1"
    if history:
        if (
            kind != "predictions"
            or value["kind"] != kind
            or value["files"] != identity["files"]
            or value["stats"] != identity["stats"]
        ):
            raise ValueError("Invalid baseline metadata")
        prefix = "history_"
        required = {"features.jsonl", "predictions.jsonl"}
    elif value["format"] == "temporal-xgb-v1":
        prefix = "tsxgb_"
        required = {
            "samples": {"samples.jsonl", "requests.jsonl", "eligibility.jsonl", "report.json"},
            "models": {"predictions.jsonl", "report.json"},
            "scores": {"scores.jsonl", "metrics.json"},
        }[kind]
    else:
        raise ValueError("Unsupported input artifact format")
    if value["artifact_id"] != prefix + digest(identity) or not required <= set(identity["files"]):
        raise ValueError("Artifact identity/inventory failure")
    inventory(root, identity["files"])
    return value


def verify(root):
    root = Path(root)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("Symlink artifact rejected")
    value = load(root / "manifest.json")
    if value.get("format") != FORMAT or digest(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    ) != value.get("manifest_sha256"):
        raise ValueError("Acceptance manifest integrity failure")
    identity = value["identity"]
    if (
        identity.get("format") != FORMAT
        or identity["kind"] not in REQUIRED
        or value["artifact_id"] != "acceptance_" + digest(identity)
        or not REQUIRED[identity["kind"]] <= set(identity["files"])
    ):
        raise ValueError("Wrong acceptance artifact identity")
    inventory(root, identity["files"])
    return value


def register(database, path):
    if not Path(database).is_file():
        raise FileNotFoundError("Existing project database required")
    value = verify(path)
    record = (
        value["artifact_id"],
        value["identity"]["kind"],
        value["manifest_sha256"],
        str(Path(path).resolve()),
    )
    with closing(sqlite3.connect(database, timeout=5)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS temporal_acceptance_artifacts(artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, path TEXT NOT NULL)"
        )
        previous = connection.execute(
            "SELECT artifact_id,kind,manifest_sha256,path FROM temporal_acceptance_artifacts WHERE artifact_id=?",
            (record[0],),
        ).fetchone()
        if previous and previous != record:
            raise ValueError("Immutable acceptance registry conflict")
        connection.execute(
            "INSERT OR IGNORE INTO temporal_acceptance_artifacts VALUES (?,?,?,?)", record
        )


class Publication:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=".acceptance-", dir=self.output))

    def close(self):
        shutil.rmtree(self.stage, ignore_errors=True)

    def publish(self, kind, inputs, database):
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(self.stage.iterdir())
            if p.is_file()
        }
        identity = {"format": FORMAT, "kind": kind, "inputs": inputs, "files": files}
        value = {
            "format": FORMAT,
            "artifact_id": "acceptance_" + digest(identity),
            "identity": identity,
        }
        value["manifest_sha256"] = digest(value)
        (self.stage / "manifest.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
        verify(self.stage)
        final = self.output / value["artifact_id"]
        if final.exists():
            if verify(final) != value:
                raise ValueError("Acceptance artifact conflict")
            register(database, final)
            self.close()
            return {**value, "artifact_path": str(final), "reused": True}
        self.stage.rename(final)
        try:
            register(database, final)
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            raise
        return {**value, "artifact_path": str(final), "reused": False}
