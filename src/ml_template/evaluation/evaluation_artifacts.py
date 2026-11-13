"""Strict JSON and SHA-256 integrity for evaluation artifacts."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if x is pd.NA or x is pd.NaT or isinstance(x, float) and not math.isfinite(x):
        return None
    return x if x is None or isinstance(x, (str, int, float, bool)) else str(x)


def write_json(path, data):
    Path(path).write_text(
        json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(directory):
    directory = Path(directory)
    entries = {
        p.relative_to(directory).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    write_json(directory / "artifact_manifest.json", {"files": entries})


def verify(directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory / "artifact_manifest.json").read_text())
    for name, item in meta["files"].items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory) or sha256(p) != item["sha256"]:
            raise ValueError(f"artifact differs from manifest: {name}")
    if "report_data.json" not in meta["files"]:
        raise ValueError("missing report_data manifest entry")
    return json.loads((directory / "report_data.json").read_text())
