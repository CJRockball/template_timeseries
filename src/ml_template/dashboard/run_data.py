# threshold-diagnostics-v1
"""Loaders for the single-run page. Reads project.db and run folders; no Dash or Plotly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from ml_template.paths import ARTIFACTS_RUNS, DATA_SILVER

RUN_COLUMNS = (
    "run_id, experiment_name, experiment_run_id, seed, feature_set_id, model_type, silver_dataset_version_id, "
    "primary_metric, primary_value, n_rounds_final, n_features, n_rows, duration_s, "
    "started_at, finished_at, holdout_evaluated_at, artifact_dir, split_id, git_sha, git_dirty"
)


def _query(engine: Engine, sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def list_runs(
    engine: Engine, experiment_name: str | None = None, status: str = "succeeded"
) -> pd.DataFrame:
    """Runs, newest first. ISO UTC strings sort correctly as text."""
    sql = f"SELECT {RUN_COLUMNS} FROM model_runs WHERE status = :status"
    params: dict = {"status": status}
    if experiment_name:
        sql += " AND experiment_name = :name"
        params["name"] = experiment_name
    return _query(engine, sql + " ORDER BY started_at DESC", **params)


def latest_run_id(engine: Engine, experiment_name: str | None = None) -> str | None:
    runs = list_runs(engine, experiment_name)
    return None if runs.empty else str(runs["run_id"].iloc[0])


def run_header(engine: Engine, run_id: str) -> dict:
    df = _query(engine, f"SELECT {RUN_COLUMNS} FROM model_runs WHERE run_id = :r", r=run_id)
    if df.empty:
        raise KeyError(f"unknown run_id: {run_id}")
    return df.iloc[0].to_dict()


def load_params(engine: Engine, run_id: str) -> dict[str, str]:
    df = _query(engine, "SELECT key, value FROM run_params WHERE run_id = :r", r=run_id)
    return dict(zip(df["key"], df["value"], strict=True))


def load_metrics(engine: Engine, run_id: str) -> pd.DataFrame:
    """Columns: split, fold, metric, value. Fold -1 is pooled."""
    return _query(
        engine,
        "SELECT split, fold, metric, value FROM run_metrics WHERE run_id = :r",
        r=run_id,
    )


def load_importance(engine: Engine, run_id: str, fold: int = -1) -> pd.DataFrame:
    """Columns: method, feature, value, std. Fold -1 is the mean over folds."""
    return _query(
        engine,
        "SELECT method, feature, value, std FROM run_feature_importance "
        "WHERE run_id = :r AND fold = :f ORDER BY value DESC",
        r=run_id,
        f=fold,
    )


def load_labels(engine: Engine, header: dict, silver_root: Path = DATA_SILVER) -> pd.Series | None:
    """Version-checked Silver labels with canonical unique keys; never truncate fractional labels."""
    from ml_template.dashboard.regression_data import guard_binary_labels

    guard_binary_labels(header)
    split = _query(engine, "SELECT * FROM splits WHERE split_id = :s", s=header["split_id"])
    if split.empty:
        return None
    row = split.iloc[0]
    key, target = (row["key_col"], row["target_col"])
    if pd.isna(key) or pd.isna(target) or (not key) or (not target):
        return None
    silver_id = header["silver_dataset_version_id"]
    if row["silver_dataset_version_id"] != silver_id:
        raise ValueError("Stored split and run reference different Silver versions")
    matches = sorted(Path(silver_root).glob(f"*/{silver_id}/data.parquet"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous Silver version: multiple datasets match the run")
    df = pd.read_parquet(matches[0], columns=[key, target])
    if df[key].isna().any() or df[key].duplicated().any():
        raise ValueError("Silver entity keys must be unique and non-null")
    if not df[target].isin([0, 1]).all():
        raise ValueError("Silver target must contain only non-null binary 0/1 labels")
    return df.set_index(key)[target].astype(int)


def load_oof_with_labels(
    engine: Engine, header: dict, folder: Path, silver_root: Path = DATA_SILVER
) -> pd.DataFrame | None:
    """Strict full-development OOF join. Missing files return None; invalid data raises.

    No rows are dropped, and every row/fold must match the stored development partition.
    """
    from ml_template.dashboard.compare_data import align_predictions, attach_labels
    from ml_template.paths import DATA_SPLITS, ROOT

    oof = load_oof(folder)
    if oof is None:
        return None
    labels = load_labels(engine, header, silver_root)
    if labels is None:
        return None
    split = _query(engine, "SELECT * FROM splits WHERE split_id = :s", s=header["split_id"])
    if split.empty:
        return None
    row = split.iloc[0]
    stored = row.get("path")
    path = (
        Path(str(stored))
        if stored is not None and not pd.isna(stored)
        else DATA_SPLITS / header["split_id"]
    )
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path = path / "assignments.parquet"
    if not path.is_file():
        path = DATA_SPLITS / str(header["split_id"]) / "assignments.parquet"
    if not path.is_file():
        return None
    assignments = pd.read_parquet(path)
    required = {labels.index.name, "partition", "fold"}
    if not required.issubset(assignments):
        raise ValueError("Stored split assignments lack key/partition/fold columns")
    if not assignments["partition"].isin(["dev", "holdout"]).all():
        raise ValueError("Stored split has invalid partition labels")
    dev_folds = pd.to_numeric(
        assignments.loc[assignments["partition"] == "dev", "fold"], errors="coerce"
    )
    if not dev_folds.map(
        lambda value: pd.notna(value) and value >= 0 and float(value).is_integer()
    ).all():
        raise ValueError("Stored development folds must be nonnegative integers")
    raw = attach_labels(oof, labels, str(labels.index.name))
    return align_predictions({header["run_id"]: raw}, str(labels.index.name), assignments)[
        header["run_id"]
    ]


def run_dir(header: dict, base: Path = ARTIFACTS_RUNS) -> Path:
    stored = header.get("artifact_dir")
    if stored and Path(stored).is_dir():
        return Path(stored)
    return base / str(header["run_id"])


def read_artifact(folder: Path, name: str) -> pd.DataFrame | None:
    """CSV or Parquet by suffix; None when the file was not written for this run."""
    path = folder / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_learning_curves(folder: Path) -> pd.DataFrame | None:
    """Columns: fold, round, dataset, metric, value, best_round."""
    return read_artifact(folder, "learning_curves.parquet")


def load_cv_folds(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "cv_folds.csv")


def load_error_sample(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "error_sample.csv")


def load_oof(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "oof_predictions.parquet")


def load_config(folder: Path) -> dict | None:
    path = folder / "config.json"
    return json.loads(path.read_text()) if path.exists() else None
