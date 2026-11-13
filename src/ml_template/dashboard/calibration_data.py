"""Read-only calibration dashboard data. No fitting, migrations or holdout access."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("raw", "calibrated_crossfit")
SPLIT = "dev_calibration_diagnostic"
RUN_COLUMNS = [
    "calibration_id",
    "parent_run_id",
    "status",
    "started_at",
    "finished_at",
    "method",
    "config_json",
    "artifact_dir",
    "error_summary",
]
ARTIFACTS = (
    "config.json",
    "provenance.json",
    "calibrator.json",
    "diagnostic_notes.json",
    "metrics.csv",
    "reliability.csv",
    "development_predictions.parquet",
)


def _query(engine, sql, params=None):
    if isinstance(engine, sqlite3.Connection):
        return pd.read_sql_query(sql, engine, params=params or {})
    from sqlalchemy import text

    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection, params=params or {})


def _has_table(engine, name):
    if isinstance(engine, sqlite3.Connection):
        return (
            engine.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            is not None
        )
    from sqlalchemy import inspect

    return inspect(engine).has_table(name)


def list_calibrations(engine, parent_run_id=None):
    """Newest first; missing optional tables stay missing (no create_all calls)."""
    if not _has_table(engine, "calibration_runs"):
        return pd.DataFrame(columns=RUN_COLUMNS)
    sql = "SELECT " + ", ".join(RUN_COLUMNS) + " FROM calibration_runs"
    params = {}
    if parent_run_id:
        sql += " WHERE parent_run_id = :parent"
        params["parent"] = parent_run_id
    return _query(engine, sql + " ORDER BY started_at DESC, calibration_id DESC", params)


def calibration_header(engine, calibration_id):
    if not _has_table(engine, "calibration_runs"):
        raise KeyError("No calibration_runs table; run the calibration CLI first")
    table = _query(
        engine,
        "SELECT " + ", ".join(RUN_COLUMNS) + " FROM calibration_runs WHERE calibration_id = :id",
        {"id": calibration_id},
    )
    if len(table) != 1:
        raise KeyError(f"Unknown calibration ID: {calibration_id}")
    return table.iloc[0].to_dict()


def _root(repo_root):
    if repo_root is not None:
        return Path(repo_root).resolve()
    from ml_template.paths import ROOT

    return ROOT


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _folder(record, root):
    cid = str(record["calibration_id"])
    if not re.fullmatch(r"cal_[A-Za-z0-9_-]+", cid):
        raise ValueError("Invalid calibration ID")
    stored = record.get("artifact_dir")
    candidate = Path(str(stored)) if stored else root / "artifacts" / "calibration" / cid
    if not candidate.is_absolute():
        candidate = root / candidate
    fallback = root / "artifacts" / "calibration" / cid
    if candidate.is_dir():
        return candidate.resolve(), None
    if fallback.is_dir():
        return (
            fallback.resolve(),
            "Recorded directory is unavailable; using relocated calibration directory",
        )
    return candidate.resolve(), None


def validate_metrics(table):
    required = {"split", "variant", "fold", "metric", "value"}
    if not required.issubset(table.columns):
        raise ValueError("metrics.csv has missing columns")
    out = table.loc[table["split"].eq(SPLIT) & table["variant"].isin(VARIANTS)].copy()
    out["fold"] = pd.to_numeric(out["fold"], errors="raise")
    if not np.isfinite(out["fold"]).all() or (out["fold"] % 1 != 0).any():
        raise ValueError("metric fold IDs must be finite integers")
    out["fold"] = out["fold"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if np.isinf(out["value"]).any():
        raise ValueError("metrics contain infinite values")
    if out.duplicated(["split", "variant", "fold", "metric"]).any():
        raise ValueError("metrics contain duplicate identities")
    return out


def validate_reliability(table):
    required = {
        "variant",
        "bin",
        "n_rows",
        "n_positives",
        "mean_probability",
        "event_rate",
        "event_rate_low",
        "event_rate_high",
    }
    if not required.issubset(table.columns):
        raise ValueError("reliability.csv has missing columns")
    out = table.loc[table["variant"].isin(VARIANTS)].copy()
    for col in required - {"variant"}:
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col]).all():
            raise ValueError("reliability contains nonfinite values")
    if (
        (out.n_rows <= 0).any()
        or (out.n_positives < 0).any()
        or (out.n_positives > out.n_rows).any()
    ):
        raise ValueError("invalid reliability support counts")
    for col in ["n_rows", "n_positives", "bin"]:
        if (out[col] % 1 != 0).any():
            raise ValueError("reliability counts and bin IDs must be integers")
    for col in ["mean_probability", "event_rate", "event_rate_low", "event_rate_high"]:
        if not out[col].between(0, 1).all():
            raise ValueError("reliability rates must be in [0, 1]")
    if (out.event_rate_low > out.event_rate).any() or (out.event_rate_high < out.event_rate).any():
        raise ValueError("reliability intervals do not contain event rates")
    if not np.allclose(out.event_rate, out.n_positives / out.n_rows):
        raise ValueError("reliability counts and event rates disagree")
    if out.duplicated(["variant", "bin"]).any():
        raise ValueError("duplicate reliability bins")
    return out


def validate_predictions(table, key_col):
    required = {key_col, "fold", "y_true", "p_raw", "p_cal_crossfit"}
    if not required.issubset(table.columns):
        raise ValueError("development predictions have missing columns")
    # Never send p_cal_fit to the dashboard's assessment tables or charts.
    out = table[[key_col, "fold", "y_true", "p_raw", "p_cal_crossfit"]].copy()
    if out.empty or out[key_col].isna().any() or out[key_col].duplicated().any():
        raise ValueError("prediction keys must be unique and non-null")
    if not out["y_true"].isin([0, 1]).all():
        raise ValueError("prediction labels must be binary")
    folds = pd.to_numeric(out["fold"], errors="raise")
    if not np.isfinite(folds).all() or (folds < 0).any() or (folds % 1 != 0).any():
        raise ValueError("predictions must contain development folds only")
    out["fold"] = folds.astype(int)
    for col in ["p_raw", "p_cal_crossfit"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col]).all() or not out[col].between(0, 1).all():
            raise ValueError("invalid development probabilities")
    return out


def load_calibration(engine, calibration_id, *, repo_root=None, include_predictions=True):
    """Read whitelisted saved artifacts; suppress unverified or malformed sections."""
    record = calibration_header(engine, calibration_id)
    folder, relocated = _folder(record, _root(repo_root))
    bundle = {
        "header": record,
        "folder": str(folder),
        "warnings": [],
        "inventory": pd.DataFrame(),
        "config": {},
        "provenance": {},
        "calibrator": {},
        "notes": {},
        "metrics": pd.DataFrame(),
        "reliability": pd.DataFrame(),
        "predictions": pd.DataFrame(),
    }
    warnings = bundle["warnings"]
    if relocated:
        warnings.append(relocated)
    if record["status"] != "succeeded":
        warnings.append(
            f"Calibration status is {record['status']}; completed diagnostics are not loaded"
        )
        if record.get("error_summary"):
            warnings.append(str(record["error_summary"]))
        return bundle
    try:
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        manifest_files = manifest["files"]
        if not isinstance(manifest_files, dict):
            raise ValueError("invalid files mapping")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"Cannot verify calibration artifacts: {exc}")
        return bundle
    inventory = []
    verified = set()
    for name in ARTIFACTS:
        path = folder / name
        state = "missing"
        expected = manifest_files.get(name)
        if path.is_file():
            if not isinstance(expected, dict) or "sha256" not in expected:
                state = "not in manifest"
            elif _sha256(path) != expected["sha256"]:
                state = "hash mismatch"
            else:
                state = "verified"
                verified.add(name)
        inventory.append({"artifact": name, "state": state})
        if state != "verified":
            warnings.append(f"{name}: {state}; section unavailable")
    bundle["inventory"] = pd.DataFrame(inventory)
    for name, field in [
        ("config.json", "config"),
        ("provenance.json", "provenance"),
        ("calibrator.json", "calibrator"),
        ("diagnostic_notes.json", "notes"),
    ]:
        if name in verified:
            try:
                obj = json.loads((folder / name).read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    raise ValueError("expected a JSON object")
                bundle[field] = obj
            except (OSError, ValueError) as exc:
                warnings.append(f"{name}: {exc}")
    for field in ["config", "provenance"]:
        if bundle[field].get("parent_run_id") != record["parent_run_id"]:
            warnings.append(
                f"{field} parent binding unavailable or mismatched; all diagnostic data suppressed"
            )
            bundle["calibrator"] = {}
            return bundle
    if bundle["config"].get("method") != record["method"]:
        warnings.append("Configuration method disagrees with registry; diagnostic data suppressed")
        bundle["calibrator"] = {}
        return bundle
    for name, field, validator in [
        ("metrics.csv", "metrics", validate_metrics),
        ("reliability.csv", "reliability", validate_reliability),
    ]:
        if name in verified:
            try:
                bundle[field] = validator(pd.read_csv(folder / name))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"{name}: {exc}")
    if include_predictions and "development_predictions.parquet" in verified:
        try:
            key = bundle["provenance"]["key_col"]
            # Project only approved diagnostic columns, never p_cal_fit.
            columns = [key, "fold", "y_true", "p_raw", "p_cal_crossfit"]
            bundle["predictions"] = validate_predictions(
                pd.read_parquet(folder / "development_predictions.parquet", columns=columns), key
            )
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            warnings.append(f"development_predictions.parquet: {exc}")
    if bundle["notes"].get("warning_slope_at_floor"):
        warnings.append(
            "Final sigmoid slope is at its lower bound: investigate weak/reversed score signal"
        )
    if bundle["provenance"].get("parent_holdout_evaluated_at"):
        warnings.append(
            "Parent holdout had already been evaluated before calibration; results are not a fresh final test"
        )
    return bundle


def metric_comparison(metrics, fold=-1):
    columns = ["metric", "raw", "calibrated_crossfit", "delta_cal_minus_raw", "direction"]
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=columns)
    part = metrics.loc[metrics["fold"].eq(fold) & metrics["variant"].isin(VARIANTS)]
    if part.empty:
        return pd.DataFrame(columns=columns)
    wide = (
        part.pivot(index="metric", columns="variant", values="value")
        .reindex(columns=list(VARIANTS))
        .reset_index()
    )
    wide["delta_cal_minus_raw"] = wide.calibrated_crossfit - wide.raw
    wide["direction"] = (
        wide["metric"]
        .map(
            {
                "brier": "lower is better",
                "log_loss": "lower is better",
                "pr_auc": "higher is better",
                "roc_auc": "higher is better",
            }
        )
        .fillna("descriptive")
    )
    return wide[columns]


def distribution_table(predictions, n_bins=30):
    columns = ["variant", "class", "bin_left", "bin_right", "bin_mid", "count"]
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=columns)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for variant, column in [("raw", "p_raw"), ("calibrated_crossfit", "p_cal_crossfit")]:
        for label in [0, 1]:
            counts, _ = np.histogram(
                predictions.loc[predictions.y_true.eq(label), column], bins=edges
            )
            for i, count in enumerate(counts):
                rows.append(
                    {
                        "variant": variant,
                        "class": label,
                        "bin_left": edges[i],
                        "bin_right": edges[i + 1],
                        "bin_mid": (edges[i] + edges[i + 1]) / 2,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def metadata_table(bundle):
    values = {
        k: bundle["header"].get(k)
        for k in [
            "calibration_id",
            "parent_run_id",
            "status",
            "method",
            "started_at",
            "finished_at",
        ]
    }
    values.update(
        {
            k: bundle["provenance"].get(k)
            for k in [
                "silver_dataset_version_id",
                "feature_set_id",
                "split_id",
                "key_col",
                "target_col",
                "parent_holdout_evaluated_at",
            ]
        }
    )
    values["artifact_directory"] = bundle["folder"]
    return pd.DataFrame([{"field": k, "value": v} for k, v in values.items()])


def threshold_inputs(bundle, score_variant="calibrated_crossfit"):
    """Adapt only verified, keyed development scores to shared threshold panels."""
    if score_variant not in VARIANTS:
        raise ValueError("unsupported threshold score variant")
    frame = bundle["predictions"]
    if frame is None or frame.empty:
        return {}, {}
    key = bundle["provenance"]["key_col"]
    frame = validate_predictions(frame, key).sort_values(key, kind="stable")
    column = "p_raw" if score_variant == "raw" else "p_cal_crossfit"
    view = frame[[key, "fold", "y_true", column]].rename(columns={column: "p_oof"})
    label = "Raw OOF" if score_variant == "raw" else "Calibrated cross-fit"
    metadata = {
        label: {
            "parent_run_id": bundle["header"]["parent_run_id"],
            "calibration_id": bundle["header"]["calibration_id"],
            "score_variant": score_variant,
            "source_score_column": column,
        }
    }
    return {label: view}, metadata


# calibrated-policy update v1
