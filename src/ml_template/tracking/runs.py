"""Lightweight run tracking: one folder per run, config + git + data state.

Deliberately not MLflow. For tens of runs, a folder per run plus an
append-only index is enough — every artifact stays diffable plain text
and there is no server to maintain.

Design rule: this module must stay project-agnostic (the only file
dependency is the ARTIFACTS default). tracking.py + paths.py + utils.py
are the reusable kernel for future projects.

Layout
------
artifacts/runs/
├── index.csv                      # one row per run, appended
└── 20260826_090000_sel_g025/
    ├── config.json                # args, constants, hyperparams, versions, git sha, note
    ├── dataset.json               # input identity: hashes, shape, target stats
    ├── splits.json                # positional train/holdout indices
    ├── features.json              # final feature list + selection stability
    ├── oof.npy
    ├── cv_metrics.parquet
    ├── summary_oof.csv / summary_hold.csv
    ├── pr_curve_holdout.png / conf_heatmap.png
    └── shap/ ...
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ml_template.paths import ARTIFACTS

TRACKED_PKGS = ("numpy", "pandas", "scikit-learn", "xgboost")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> dict:
    out = {}
    for pkg in TRACKED_PKGS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _slugify(s: str) -> str:
    """Filesystem-safe run name: lowercase, [a-z0-9._-], dash-separated."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-").lower()
    return s or "run"


def hash_file(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Content hash of a file, streamed in chunks. Byte-identity: answers
    'did this file change?', not 'did the values change?' (parquet rewrites
    change bytes even for identical data — that is the intended semantics)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_run(
    config: dict, run_name: str | None = None, note: str = "", runs_root: Path | None = None
) -> tuple[Path, dict]:
    """Create artifacts/runs/<timestamp>_<name>/ and write config.json."""
    runs_root = runs_root or ARTIFACTS / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{ts}_{_slugify(run_name or 'run')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "run_id": run_dir.name,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "packages": _pkg_versions(),
        "note": note,
        "config": config,
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    return run_dir, meta


def save_splits(run_dir: Path, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Persist the exact split. Positional indices are meaningful because
    load_data sorts by timestamp and resets the index."""
    payload = {
        "train_index": df_train.index.tolist(),
        "holdout_index": df_test.index.tolist(),
        "n_train_fails": int(df_train["is_fail"].sum()),
        "n_holdout_fails": int(df_test["is_fail"].sum()),
    }
    (run_dir / "splits.json").write_text(json.dumps(payload))


def save_features(run_dir: Path, feats: list, stability: pd.Series | None = None) -> None:
    payload = {"n_features": len(feats), "features": [str(f) for f in feats]}
    if stability is not None:
        payload["stability"] = {str(k): float(v) for k, v in stability.items()}
    (run_dir / "features.json").write_text(json.dumps(payload, indent=2))


def append_index(run_dir: Path, metrics: dict, index_file: Path | None = None) -> None:
    index_file = index_file or ARTIFACTS / "index.csv"
    row = pd.DataFrame([{"run_id": run_dir.name, **metrics}])
    if index_file.exists():
        row = pd.concat([pd.read_csv(index_file), row], ignore_index=True)
    row.to_csv(index_file, index=False)


def write_dataset_card(
    parquet_path: Path, df: pd.DataFrame, target: pd.Series | None = None
) -> Path:
    """Sidecar card, written by the PRODUCER at dataset creation."""
    info = {
        "file": parquet_path.name,
        "sha256_16": hash_file(parquet_path)[:16],
        "n_rows": len(df),
        "n_features": int(df.shape[1]),
        "columns_md5_16": hashlib.md5("|".join(map(str, df.columns)).encode()).hexdigest()[:16],
    }
    if target is not None:
        info.update(n_fails=int(target.sum()), prevalence=float(target.mean()))
    card = parquet_path.with_suffix(".dataset.json")  # dfX_v2.dataset.json
    card.write_text(json.dumps(info, indent=2))
    return card
