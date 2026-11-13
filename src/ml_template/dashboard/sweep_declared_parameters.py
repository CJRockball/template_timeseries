"""Use sweep YAML grid keys to select recorded parameters for a whole launch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PARAM_PREFIX = "param::"


def sweep_grid(document: object) -> dict:
    """Validate the existing runner's base/grid sweep schema without executing it."""
    if not isinstance(document, dict):
        raise ValueError("Sweep YAML must contain a mapping")
    unknown = set(document) - {"base", "experiment_name", "grid"}
    if unknown:
        raise ValueError(f"Unsupported sweep YAML keys: {sorted(map(str, unknown))}")
    if not isinstance(document.get("base"), str) or not document["base"].strip():
        raise ValueError("Sweep YAML requires a base experiment path")
    grid = document.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Sweep YAML requires a non-empty grid mapping")
    for key, values in grid.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError("Sweep grid keys must be non-empty configuration-key strings")
        if key.startswith(PARAM_PREFIX):
            raise ValueError("Sweep grid keys must not include the storage prefix param::")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep grid {key!r} requires a non-empty value list")
    return grid


def read_sweep_document(path: str | Path, *, repo_root: Path) -> tuple[dict, Path]:
    """Match runner path precedence: absolute, existing cwd-relative, then repo root.

    Read only the sweep document, never its base experiment/dataset YAMLs.
    The current runner stores a path, not an immutable launch-time YAML snapshot.
    """
    if path is None or pd.isna(path) or not str(path).strip():
        raise ValueError("Selected launch has no recorded sweep_path")
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = Path(repo_root) / candidate
    if not candidate.is_file():
        raise ValueError(f"Recorded sweep YAML not found: {path}")
    try:
        with candidate.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid sweep YAML: {candidate}: {error}") from error
    sweep_grid(document)
    return document, candidate.resolve()


def declared_parameter_frame(
    runs: pd.DataFrame, params: pd.DataFrame, document: dict, *, launch_id: str
) -> pd.DataFrame:
    """Keep every run in one experiment_run_id and ONLY grid-declared parameters.

    Parameter values come from run_params, not the YAML candidate lists. Include
    seed/analysis keys when explicitly declared: do not reuse parameter_frame's
    general-purpose exclusions. Never restrict other swept settings to one value.
    """
    grid = sweep_grid(document)
    if "experiment_run_id" not in runs or "run_id" not in runs:
        raise ValueError("Run inventory needs experiment_run_id and run_id")
    selected = runs.loc[runs["experiment_run_id"].eq(launch_id)].copy()
    selected = selected.drop(columns=[c for c in selected if str(c).startswith(PARAM_PREFIX)])
    if selected["run_id"].isna().any() or selected["run_id"].duplicated().any():
        raise ValueError("Expected one row per non-null run_id")
    needed = {"run_id", "key", "value"}
    if not needed.issubset(params.columns):
        raise ValueError("Parameter inventory needs run_id, key and value")
    part = params.loc[params["run_id"].isin(selected["run_id"]) & params["key"].isin(grid)]
    if part.duplicated(["run_id", "key"]).any():
        raise ValueError("Duplicate recorded sweep parameter keys")
    if not part.empty:
        wide = part.pivot(index="run_id", columns="key", values="value")
        wide.columns = [PARAM_PREFIX + str(key) for key in wide.columns]
        selected = selected.merge(wide, on="run_id", how="left", validate="one_to_one")
    for key in grid:
        column = PARAM_PREFIX + key
        if column not in selected:
            selected[column] = None
    return selected


def load_launch_summary_inputs(engine, launch_id: str):
    """Read registry, current sweep YAML and all launch run snapshots; no writes."""
    from ml_template.dashboard import sweep_data as sd
    from ml_template.paths import ROOT

    launches, runs, params = sd.load_sweep_data(engine)
    launch = launches.loc[launches["experiment_run_id"].eq(launch_id)]
    if len(launch) != 1:
        raise ValueError("Select exactly one registered sweep launch")
    record = launch.iloc[0]
    document, source = read_sweep_document(record.get("sweep_path"), repo_root=ROOT)
    name = document.get("experiment_name")
    if name is not None and name != record.get("experiment_name"):
        raise ValueError("Current sweep YAML experiment_name differs from the recorded launch")
    frame = declared_parameter_frame(runs, params, document, launch_id=launch_id)
    return frame, document, str(source)
