"""Experiment queries and seed summaries. No Dash imports or database writes."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import Engine

COHORT = ["silver_dataset_version_id", "split_id", "primary_metric"]
EXCLUDED = {"seed", "experiment_name", "schema_version"}


def load_experiment_runs(engine: Engine) -> pd.DataFrame:
    """Read all statuses. A refresh/reload executes these queries again."""
    from sqlalchemy import text

    with engine.connect() as conn:
        runs = pd.read_sql(text("SELECT * FROM model_runs"), conn)
        params = pd.read_sql(text("SELECT run_id, key, value FROM run_params"), conn)
    return prepare_runs(runs, params)


def prepare_runs(runs: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Group equal training recipes; retain split.seed and split.cv_seed.

    Missing parameter snapshots stay isolated, never guessed from model_type.
    A recipe is configuration identity, not proof that dirty code is identical.
    """
    out = runs.copy()
    snapshots = {}
    if not params.empty:
        if params.duplicated(["run_id", "key"]).any():
            raise ValueError("Duplicate run parameter keys")
        snapshots = {
            run_id: dict(zip(part["key"], part["value"], strict=True))
            for run_id, part in params.groupby("run_id")
        }
    recipes, recorded = [], []
    for row in out.itertuples():
        snapshot = snapshots.get(row.run_id, {})
        recorded.append(bool(snapshot))
        if not snapshot:
            recipes.append(f"unverified:{row.run_id}")
            continue
        kept = {
            key: value
            for key, value in snapshot.items()
            if key not in EXCLUDED and not key.startswith(("analysis.", "metrics."))
        }
        # Includes feature set, preprocessing, model params and trainer settings.
        # Keep recorded code revision separate as well as the training recipe.
        payload = [kept, str(getattr(row, "git_sha", None)), str(getattr(row, "git_dirty", None))]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        recipes.append(digest[:16])
    out["recipe_id"] = recipes
    out["params_recorded"] = recorded
    return out


def cohort_options(runs: pd.DataFrame) -> list[dict]:
    if runs.empty:
        return []
    values = runs[COHORT].dropna().drop_duplicates().sort_values(COHORT)
    return [
        {"label": " | ".join(map(str, row)), "value": json.dumps(list(row))}
        for row in values.itertuples(index=False, name=None)
    ]


def select_cohort(runs: pd.DataFrame, value: str | None) -> pd.DataFrame:
    if value is None:
        return runs.iloc[:0].copy()
    values = json.loads(value)
    if len(values) != len(COHORT):
        raise ValueError("Invalid cohort")
    mask = pd.Series(True, index=runs.index)
    for column, selected in zip(COHORT, values, strict=True):
        mask &= runs[column].eq(selected)
    return runs.loc[mask].copy()


def experiment_inventory(runs: pd.DataFrame) -> pd.DataFrame:
    """Status counts, including launches without any successful model yet."""
    columns = ["experiment_name", "runs", "succeeded", "running", "failed", "latest_sgt"]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    table = runs.groupby("experiment_name").agg(
        runs=("run_id", "size"), latest_sgt=("started_at", "max")
    )
    counts = pd.crosstab(runs["experiment_name"], runs["status"])
    for status in ("succeeded", "running", "failed"):
        table[status] = counts.get(status, 0)
    table["latest_sgt"] = (
        pd.to_datetime(table["latest_sgt"], utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    return table.reset_index()[columns].sort_values("latest_sgt", ascending=False)


def seed_rows(runs: pd.DataFrame) -> pd.DataFrame:
    """Latest scored successful run per seed and recipe. No holdout metrics.

    Repeat launches with the same seed do not increase the seed sample size.
    Null seeds are retained separately and excluded from sample-size statistics.
    """
    out = runs.loc[(runs["status"] == "succeeded") & runs["primary_value"].notna()].copy()
    if out.empty:
        return out
    out["_seed"] = out["seed"].astype("string")
    missing = out["seed"].isna()
    out.loc[missing, "_seed"] = "unrecorded:" + out.loc[missing, "run_id"].astype(str)
    out = out.sort_values(["started_at", "run_id"], ascending=[False, False])
    return out.drop_duplicates([*COHORT, "experiment_name", "recipe_id", "_seed"]).drop(
        columns="_seed"
    )


def seed_leaderboard(runs: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """One row per named experiment + recipe, within exactly one cohort.

    std is sample standard deviation across model seeds, not a confidence interval.
    """
    columns = [
        "rank",
        "experiment_name",
        "recipe_id",
        "n_seeds",
        "mean",
        "std",
        "min",
        "max",
        "median_duration_s",
        "n_scored_runs",
        "excluded_repeats",
        "params_recorded",
    ]
    if runs[COHORT].drop_duplicates().shape[0] > 1:
        raise ValueError("Select one dataset, split and metric before ranking")
    usable = runs[(runs["status"] == "succeeded") & runs["primary_value"].notna()]
    all_rows = seed_rows(runs)
    rows = all_rows[all_rows["seed"].notna()]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    keys = ["experiment_name", "recipe_id"]
    result = rows.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        mean=("primary_value", "mean"),
        std=("primary_value", "std"),
        min=("primary_value", "min"),
        max=("primary_value", "max"),
        median_duration_s=("duration_s", "median"),
        params_recorded=("params_recorded", "all"),
    )
    result["n_scored_runs"] = usable.groupby(keys).size()
    retained = all_rows.groupby(keys).size()
    result["excluded_repeats"] = result["n_scored_runs"] - retained
    result = result.reset_index().sort_values(
        ["mean", "experiment_name", "recipe_id"],
        ascending=[not higher_is_better, True, True],
        kind="stable",
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result[columns]
