"""Pure comparison tables plus lazy, read-only loading of run artifacts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import Engine

COHORT = ["silver_dataset_version_id", "split_id", "primary_metric"]


def selected_runs(runs: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("Select at least two different runs")
    if not set(run_ids).issubset(set(runs["run_id"])):
        raise ValueError("A selected run is no longer in this cohort")
    out = runs.set_index("run_id").loc[run_ids].reset_index()
    if out[COHORT].isna().any().any() or out[COHORT].drop_duplicates().shape[0] != 1:
        raise ValueError("Runs must share Silver version, split and primary metric")
    if not out["status"].eq("succeeded").all():
        raise ValueError("Only successful runs can be compared")
    return out


def attach_labels(oof: pd.DataFrame, labels: pd.Series, key: str) -> pd.DataFrame:
    if key not in oof or not {"fold", "p_oof"}.issubset(oof):
        raise ValueError("OOF artifact lacks key, fold or p_oof")
    if labels.index.has_duplicates or labels.index.isna().any():
        raise ValueError("Silver labels must have unique, non-null keys")
    if oof[key].duplicated().any() or oof[key].isna().any():
        raise ValueError("OOF keys must be unique and non-null")
    out = oof[[key, "fold", "p_oof"]].copy()
    out["y_true"] = labels.reindex(out[key]).to_numpy()
    return out


def align_predictions(
    tables: dict[str, pd.DataFrame], key: str, assignments: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Fail closed: no inner joins, dropped labels, duplicate keys or mismatched folds."""
    aligned = {}
    reference = None
    for run_id, table in tables.items():
        if table.empty or not {key, "fold", "y_true", "p_oof"}.issubset(table):
            raise ValueError(f"{run_id}: incomplete OOF table")
        if table[key].duplicated().any() or table[key].isna().any():
            raise ValueError(f"{run_id}: duplicate or null OOF keys")
        if not table["y_true"].isin([0, 1]).all():
            raise ValueError(f"{run_id}: labels are missing or nonbinary")
        probability = pd.to_numeric(table["p_oof"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{run_id}: probabilities must be finite and in [0, 1]")
        folds = pd.to_numeric(table["fold"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(folds).all() or ((folds < 0) | (folds != np.floor(folds))).any():
            raise ValueError(f"{run_id}: invalid development fold IDs")
        out = table[[key, "fold", "y_true", "p_oof"]].set_index(key).sort_index()
        out = out.astype({"fold": int, "y_true": int, "p_oof": float})
        if reference is not None:
            if not out.index.equals(reference.index):
                raise ValueError(
                    f"{run_id}: OOF key coverage differs; refusing intersection-only comparison"
                )
            if not out["fold"].equals(reference["fold"]):
                raise ValueError(f"{run_id}: OOF fold assignments differ")
            if not out["y_true"].equals(reference["y_true"]):
                raise ValueError(f"{run_id}: OOF labels differ")
        if assignments is not None:
            if assignments[key].duplicated().any() or assignments[key].isna().any():
                raise ValueError("Stored split has invalid keys")
            dev = assignments[assignments["partition"] == "dev"].set_index(key).sort_index()
            if not out.index.equals(dev.index) or not out["fold"].equals(dev["fold"].astype(int)):
                raise ValueError(f"{run_id}: OOF rows/folds do not match stored development split")
        reference = out
        aligned[run_id] = out.reset_index()
    return aligned


def metric_table(metrics: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    part = metrics[(metrics["split"] == "cv") & (metrics["fold"] == -1)]
    return (
        part.pivot(index="metric", columns="run_id", values="value")
        .reindex(columns=run_ids)
        .reset_index()
    )


def parameter_diff(params: dict[str, dict], only_changed: bool = True) -> pd.DataFrame:
    keys = sorted({key for snapshot in params.values() for key in snapshot})
    table = pd.DataFrame(
        {
            rid: [snapshot.get(k, "<not recorded>") for k in keys]
            for rid, snapshot in params.items()
        },
        index=keys,
    )
    if only_changed:
        table = table[table.nunique(axis=1, dropna=False) > 1]
    return table.rename_axis("parameter").reset_index()


def feature_membership(features: dict[str, list[str] | None]) -> pd.DataFrame:
    names = sorted({name for values in features.values() if values is not None for name in values})
    return pd.DataFrame(
        {
            "feature": names,
            **{
                rid: [None if values is None else name in values for name in names]
                for rid, values in features.items()
            },
        }
    )


def prediction_tables(
    aligned: dict[str, pd.DataFrame],
    baseline: str,
    mode: str,
    cutoff: float = 0.5,
    alert_rate: float = 0.05,
):
    if baseline not in aligned:
        raise ValueError("Choose a selected baseline")
    if mode not in {"threshold", "budget"}:
        raise ValueError("Unknown operating-point mode")
    if not 0 <= cutoff <= 1 or not 0 < alert_rate <= 1:
        raise ValueError("Cutoff/budget out of range")
    reference = aligned[baseline]
    key = reference.columns[0]
    y = reference["y_true"].to_numpy()
    flags, rows = {}, []
    for rid, table in aligned.items():
        p = table["p_oof"].to_numpy()
        if mode == "budget":
            k = max(1, math.ceil(round(alert_rate * len(y), 9)))
            flag = np.zeros(len(y), dtype=bool)
            flag[np.argsort(-p, kind="stable")[:k]] = True
        else:
            flag = p >= cutoff
        flags[rid] = flag
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
        rows.append(
            {
                "run_id": rid,
                "flagged": int(flag.sum()),
                "alert_rate": float(flag.mean()),
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    outcomes = reference[[key, "fold", "y_true"]].copy()
    disagreements = np.zeros(len(y), dtype=bool)
    for rid, table in aligned.items():
        outcomes[f"score::{rid}"] = table["p_oof"].to_numpy()
        outcomes[f"flag::{rid}"] = flags[rid]
        outcomes[f"score_delta::{rid}"] = table["p_oof"].to_numpy() - reference["p_oof"].to_numpy()
        disagreements |= flags[rid] != flags[baseline]
    positives = y == 1
    missed_all = positives & ~np.logical_or.reduce(list(flags.values()))
    outcomes["missed_by_all"] = missed_all
    outcomes["disagrees_with_baseline"] = disagreements
    outcomes = outcomes[missed_all | disagreements]
    return pd.DataFrame(rows), outcomes.sort_values(
        ["missed_by_all", "y_true"], ascending=False, kind="stable"
    )


def load_comparison(
    engine: Engine, selected: pd.DataFrame, *, silver_root: Path | None = None
) -> dict:
    """Lazy imports keep pure tables testable without Dash, SQLAlchemy or Parquet."""
    import json

    from sqlalchemy import text

    from ml_template.dashboard import run_data as rd
    from ml_template.paths import DATA_SPLITS, ROOT

    bundle = {
        "headers": selected,
        "params": {},
        "features": {},
        "metrics": [],
        "importance": [],
        "curves": {},
        "predictions": {},
        "warnings": [],
    }
    first = selected.iloc[0]
    assignments, split_record = None, None
    with engine.connect() as conn:
        records = pd.read_sql(
            text("SELECT * FROM splits WHERE split_id=:s"), conn, params={"s": first["split_id"]}
        )
    if len(records) == 1:
        split_record = records.iloc[0]
        if split_record["silver_dataset_version_id"] != first["silver_dataset_version_id"]:
            bundle["warnings"].append("Stored split points to a different Silver version")
            split_record = None
        else:
            from pathlib import Path

            stored = Path(str(split_record["path"]))
            stored = stored if stored.is_absolute() else ROOT / stored
            path = stored / "assignments.parquet" if stored.is_dir() else stored
            if not path.is_file():
                path = DATA_SPLITS / str(first["split_id"]) / "assignments.parquet"
            if path.is_file():
                assignments = pd.read_parquet(path)
    if assignments is None:
        bundle["warnings"].append(
            "Stored split assignments unavailable: prediction comparisons disabled"
        )
    raw = {}
    for header in selected.to_dict("records"):
        rid = header["run_id"]
        folder = rd.run_dir(header)
        bundle["params"][rid] = rd.load_params(engine, rid)
        with engine.connect() as conn:
            metrics = pd.read_sql(
                text(
                    "SELECT split, fold, metric, value FROM run_metrics "
                    "WHERE run_id=:r AND split='cv'"
                ),
                conn,
                params={"r": rid},
            )
            imp = pd.read_sql(
                text(
                    "SELECT method, feature, value, std FROM run_feature_importance "
                    "WHERE run_id=:r AND split='cv' AND fold=-1"
                ),
                conn,
                params={"r": rid},
            )
        bundle["metrics"].append(metrics.assign(run_id=rid))
        bundle["importance"].append(imp.assign(run_id=rid))
        bundle["curves"][rid] = None
        bundle["features"][rid] = None
        try:
            bundle["curves"][rid] = rd.load_learning_curves(folder)
            path = folder / "features.json"
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            features = payload.get("features") if isinstance(payload, dict) else payload
            bundle["features"][rid] = features if isinstance(features, list) else None
        except (OSError, ValueError) as exc:
            bundle["warnings"].append(f"{rid}: optional artifact unreadable: {exc}")
            bundle["features"][rid] = None
        if bundle["features"][rid] is None:
            bundle["warnings"].append(f"{rid}: input-feature manifest missing or unsupported")
        if bundle["curves"][rid] is None:
            bundle["warnings"].append(f"{rid}: diagnostic learning curves not recorded")
        if split_record is None or assignments is None:
            continue
        try:
            oof = rd.load_oof(folder)
            labels = (
                rd.load_labels(engine, header)
                if silver_root is None
                else rd.load_labels(engine, header, silver_root=silver_root)
            )
            if oof is None or labels is None:
                raise ValueError("OOF predictions or Silver labels missing")
            raw[rid] = attach_labels(oof, labels, str(split_record["key_col"]))
        except (OSError, ValueError, KeyError) as exc:
            bundle["warnings"].append(f"{rid}: predictions unavailable: {exc}")
    bundle["metrics"] = pd.concat(bundle["metrics"], ignore_index=True)
    bundle["importance"] = pd.concat(bundle["importance"], ignore_index=True)
    if len(raw) == len(selected) and split_record is not None:
        try:
            bundle["predictions"] = align_predictions(
                raw, str(split_record["key_col"]), assignments
            )
        except ValueError as exc:
            bundle["warnings"].append(f"Prediction comparison blocked: {exc}")
    return bundle
