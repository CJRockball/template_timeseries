"Separate immutable sample/model/comparison artifacts; never edits baseline files."

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path, PurePosixPath

FORMAT = "temporal-xgb-v1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_line(stream, value):
    stream.write(canonical(value) + "\n")


def rows(root, name):
    with (Path(root) / name).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify(root, kind=None):
    root = Path(root)
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("Symlink artifact rejected")
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    unsealed = {k: v for k, v in value.items() if k != "manifest_sha256"}
    if digest(unsealed) != value.get("manifest_sha256") or value.get("format") != FORMAT:
        raise ValueError("Learned artifact integrity failure")
    identity = value["identity"]
    if identity["kind"] not in {"samples", "models", "scores"} or (
        kind and identity["kind"] != kind
    ):
        raise ValueError("Wrong learned artifact kind")
    if value["artifact_id"] != "tsxgb_" + digest(identity):
        raise ValueError("Learned artifact identity mismatch")
    expected = {"manifest.json"}
    for name, info in identity["files"].items():
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError("Unsafe inventory path")
        file = root.joinpath(*p.parts)
        if (
            file.is_symlink()
            or any(x.is_symlink() for x in file.parents if x != root.parent)
            or (not file.is_file())
        ):
            raise ValueError("Unsafe/missing artifact file")
        if file.stat().st_size != info["bytes"] or file_hash(file) != info["sha256"]:
            raise ValueError("Learned artifact file integrity failure")
        expected.add(name)
    if expected != {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}:
        raise ValueError("Artifact inventory mismatch")
    required = {
        "samples": {"samples.jsonl", "requests.jsonl", "eligibility.jsonl", "report.json"},
        "models": {"predictions.jsonl", "report.json"},
        "scores": {"scores.jsonl", "metrics.json"},
    }[identity["kind"]]
    if not required <= set(identity["files"]):
        raise ValueError("Incomplete artifact")
    return value


def register(database, root):
    manifest = verify(root)
    if not Path(database).is_file():
        raise FileNotFoundError("Existing SQLite project database required")
    row = (
        manifest["artifact_id"],
        manifest["identity"]["kind"],
        manifest["manifest_sha256"],
        str(Path(root).resolve()),
    )
    with closing(sqlite3.connect(database, timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS temporal_xgb_artifacts (artifact_id TEXT PRIMARY KEY,kind TEXT,manifest_sha256 TEXT,path TEXT)"
        )
        old = conn.execute(
            "SELECT artifact_id,kind,manifest_sha256,path FROM temporal_xgb_artifacts WHERE artifact_id=?",
            (row[0],),
        ).fetchone()
        if old and old != row:
            raise ValueError("Immutable learned registry conflict")
        conn.execute("INSERT OR IGNORE INTO temporal_xgb_artifacts VALUES (?,?,?,?)", row)


class Publication:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=".tsxgb-", dir=self.root))

    def close(self):
        shutil.rmtree(self.stage, ignore_errors=True)

    def publish(self, kind, inputs, database):
        files = {
            p.relative_to(self.stage).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": file_hash(p),
            }
            for p in sorted(self.stage.rglob("*"))
            if p.is_file()
        }
        identity = {"format": FORMAT, "kind": kind, "inputs": inputs, "files": files}
        value = {"format": FORMAT, "artifact_id": "tsxgb_" + digest(identity), "identity": identity}
        value["manifest_sha256"] = digest(value)
        (self.stage / "manifest.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
        verify(self.stage, kind)
        final = self.root / value["artifact_id"]
        if final.exists():
            old = verify(final, kind)
            if old != value:
                raise ValueError("Immutable learned artifact conflict")
            register(database, final)
            self.close()
            return {**old, "artifact_path": str(final), "reused": True}
        self.stage.rename(final)
        try:
            register(database, final)
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            raise
        return {**value, "artifact_path": str(final), "reused": False}
