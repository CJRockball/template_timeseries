"""Repository paths. Single source of truth for every folder the pipeline uses."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/ml_template/paths.py -> repo root

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_SILVER = DATA / "silver"
DATA_FEATURES = DATA / "features"
DATA_SPLITS = DATA / "splits"
DATABASE = DATA / "project.db"

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_RUNS = ARTIFACTS / "runs"
LOGS = ROOT / "logs"
SQL = ROOT / "sql"
CONFIGS = ROOT / "configs"  # was "config" (singular), which does not exist
