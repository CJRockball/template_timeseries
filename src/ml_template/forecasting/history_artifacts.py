"Immutable history/baseline artifacts and their separate SQLite registry."

import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from .policy import canonical, digest

FORMAT = "temporal-history-v1"


FILES = {
    "predictions": {"features.jsonl", "predictions.jsonl"},
    "scores": {"scores.jsonl", "metrics.json"},
}


def file_hash(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def seal(value):
    value = dict(value)
    value.pop("manifest_sha256", None)
    return {**value, "manifest_sha256": digest(value)}


def verify_artifact(path, kind=None):
    path = Path(path)
    if path.is_symlink() or (path / "manifest.json").is_symlink():
        raise ValueError("Symlink artifact rejected")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    actual = manifest.get("kind")
    if actual not in FILES or (kind and actual != kind) or manifest.get("format") != FORMAT:
        raise ValueError("Wrong artifact kind/format")
    if seal(manifest)["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise ValueError("Artifact manifest integrity failure")
    if (
        set(manifest["files"]) != FILES[actual]
        or manifest["identity"]["files"] != manifest["files"]
    ):
        raise ValueError("Artifact inventory identity mismatch")
    if manifest["artifact_id"] != "history_" + digest(manifest["identity"]):
        raise ValueError("Artifact identity mismatch")
    if manifest["identity"]["kind"] != actual or manifest["identity"]["stats"] != manifest["stats"]:
        raise ValueError("Artifact metadata identity mismatch")
    for name, info in manifest["files"].items():
        file = path / name
        if (
            file.is_symlink()
            or not file.is_file()
            or file.stat().st_size != info["bytes"]
            or (file_hash(file) != info["sha256"])
        ):
            raise ValueError("Artifact file integrity failure")
    return manifest


def write_line(handle, value):
    handle.write(canonical(value) + "\n")


def register(database, path):
    manifest = verify_artifact(path)
    if not Path(database).is_file():
        raise FileNotFoundError("Existing project SQLite database required")
    row = (
        manifest["artifact_id"],
        manifest["kind"],
        manifest["manifest_sha256"],
        str(Path(path).resolve()),
        manifest["identity"]["inputs"]["plan_id"],
    )
    with closing(sqlite3.connect(database, timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS temporal_history_artifacts (artifact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, path TEXT NOT NULL, plan_id TEXT NOT NULL)"
        )
        old = conn.execute(
            "SELECT artifact_id,kind,manifest_sha256,path,plan_id FROM temporal_history_artifacts WHERE artifact_id=?",
            (row[0],),
        ).fetchone()
        if old is not None and old != row:
            raise ValueError("Immutable history registry conflict")
        conn.execute("INSERT OR IGNORE INTO temporal_history_artifacts VALUES (?,?,?,?,?)", row)


class Publication:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=".history-build-", dir=self.root))
        self.published = None

    def close(self):
        if self.stage.exists():
            shutil.rmtree(self.stage)

    def publish(self, kind, inputs, stats, database):
        inventory = {
            name: {
                "bytes": (self.stage / name).stat().st_size,
                "sha256": file_hash(self.stage / name),
            }
            for name in sorted(FILES[kind])
        }
        identity = {
            "format": FORMAT,
            "kind": kind,
            "inputs": inputs,
            "stats": stats,
            "files": inventory,
        }
        manifest = seal(
            {
                "format": FORMAT,
                "kind": kind,
                "identity": identity,
                "stats": stats,
                "files": inventory,
                "artifact_id": "history_" + digest(identity),
            }
        )
        (self.stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_artifact(self.stage, kind)
        final = self.root / manifest["artifact_id"]
        if final.exists():
            existing = verify_artifact(final, kind)
            if existing != manifest:
                raise ValueError("Immutable history artifact conflict")
            register(database, final)
            self.close()
            return {**existing, "artifact_path": str(final), "reused": True}
        self.stage.rename(final)
        try:
            register(database, final)
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            raise
        self.published = final
        return {**manifest, "artifact_path": str(final), "reused": False}
