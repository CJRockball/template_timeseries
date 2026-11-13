"""Sweep launch inventory, varied parameters and fixed-context parameter slices."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pandas as pd

from ml_template.dashboard.inventory_common import read_registry

if TYPE_CHECKING:
    from sqlalchemy import Engine

COHORT = ["silver_dataset_version_id", "split_id", "primary_metric"]
PARAM_PREFIX = "param::"
EXCLUDE = {"seed", "experiment_name", "schema_version"}


def load_sweep_data(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    from ml_template.dashboard import experiment_data as ed

    launches = read_registry(engine, "experiment_launches")
    launches = launches[launches["kind"] == "sweep"].copy()
    runs = ed.load_experiment_runs(engine)
    with engine.connect() as conn:
        params = pd.read_sql(text("SELECT run_id, key, value FROM run_params"), conn)
    return launches, runs, params


def launch_summary(launches: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for launch in launches.to_dict("records"):
        part = runs[runs["experiment_run_id"] == launch["experiment_run_id"]]
        counts = part["status"].value_counts()
        expected = launch.get("n_runs")
        expected = int(expected) if expected is not None and not pd.isna(expected) else None
        rows.append(
            {
                **launch,
                "expected_runs": expected,
                "registered_runs": len(part),
                "not_registered": max(0, expected - len(part)) if expected is not None else None,
                "succeeded_runs": int(counts.get("succeeded", 0)),
                "running_runs": int(counts.get("running", 0)),
                "failed_runs": int(counts.get("failed", 0)),
                "sum_recorded_duration_s": part["duration_s"].sum(min_count=1),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                *launches.columns,
                "expected_runs",
                "registered_runs",
                "not_registered",
                "succeeded_runs",
                "running_runs",
                "failed_runs",
                "sum_recorded_duration_s",
            ]
        )
    return pd.DataFrame(rows).sort_values("started_at", ascending=False)


def parameter_frame(runs: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Add dotted config keys as param::<key> columns, without overwriting run metadata."""
    selected = params[params["run_id"].isin(runs["run_id"])].copy()
    keep = ~selected["key"].isin(EXCLUDE) & ~selected["key"].str.startswith(
        ("analysis.", "metrics.")
    )
    selected = selected[keep]
    if selected.duplicated(["run_id", "key"]).any():
        raise ValueError("Duplicate parameter keys")
    if selected.empty:
        return runs.copy()
    wide = selected.pivot(index="run_id", columns="key", values="value")
    wide.columns = [PARAM_PREFIX + key for key in wide.columns]
    return runs.merge(wide, on="run_id", how="left", validate="one_to_one")


def varied_parameters(frame: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in frame
        if col.startswith(PARAM_PREFIX) and frame[col].nunique(dropna=False) > 1
    )


def context_groups(frame: pd.DataFrame, axes: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Only parameters on the axes may differ within a slice; code revision is also fixed.

    Missing snapshots never share a slice. Null parameter values are distinct from strings.
    """
    out = frame.copy()
    remaining = [c for c in out if c.startswith(PARAM_PREFIX) and c not in axes]
    fingerprints, descriptions = [], {}
    for row in out.to_dict("records"):
        if not row.get("params_recorded", False):
            identity = f"unrecorded:{row['run_id']}"
            description = identity
        else:
            payload = {c: None if pd.isna(row[c]) else str(row[c]) for c in remaining}
            payload["git_sha"] = str(row.get("git_sha"))
            payload["git_dirty"] = str(row.get("git_dirty"))
            payload["model_type"] = str(row.get("model_type"))
            payload["feature_set_id"] = str(row.get("feature_set_id"))
            encoded = json.dumps(payload, sort_keys=True)
            identity = hashlib.sha256(encoded.encode()).hexdigest()[:16]
            varying_nuisance = [c for c in remaining if out[c].nunique(dropna=False) > 1]
            description = (
                ", ".join(f"{c.removeprefix(PARAM_PREFIX)}={row[c]}" for c in varying_nuisance)
                or "other parameters fixed"
            )
        fingerprints.append(identity)
        descriptions[identity] = description
    out["context_id"] = fingerprints
    options = [
        {"value": identity, "label": f"{identity[:8]} | {description}"}
        for identity, description in sorted(descriptions.items())
    ]
    return out, options


def cell_summary(frame: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """Means/std over distinct recorded model seeds, never pooled across nuisance contexts."""
    if not 1 <= len(axes) <= 2 or len(set(axes)) != len(axes):
        raise ValueError("Choose one or two different parameter axes")
    if frame["context_id"].nunique() > 1:
        raise ValueError("Select one fixed-parameter context")
    if frame[COHORT].drop_duplicates().shape[0] > 1:
        raise ValueError("Select one dataset, split and metric")
    usable = frame[
        (frame["status"] == "succeeded") & frame["primary_value"].notna() & frame["seed"].notna()
    ].copy()
    usable = usable.dropna(subset=axes)
    usable = usable.sort_values(["started_at", "run_id"], ascending=False)
    usable = usable.drop_duplicates([*axes, "seed"])
    return (
        usable.groupby(axes, dropna=False)
        .agg(
            mean=("primary_value", "mean"),
            std=("primary_value", "std"),
            n_seeds=("seed", "nunique"),
            min=("primary_value", "min"),
            max=("primary_value", "max"),
            median_duration_s=("duration_s", "median"),
        )
        .reset_index()
    )
