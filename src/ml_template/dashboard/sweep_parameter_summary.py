"""Run-weighted marginal score summaries and top-k parameter concentration.

Read-only: no database, Dash, training or holdout access. Parameter values retain
recorded string identities; missing cells are excluded, never made into categories.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

PARAM_PREFIX = "param::"
COHORT = ["silver_dataset_version_id", "split_id", "primary_metric"]
OVERALL_COLUMNS = ["parameter", "value", "n_runs", "mean", "std", "min", "max", "status"]
STABILITY_COLUMNS = [
    "parameter",
    "n_runs",
    "missing_runs",
    "distinct_values",
    "mode_values",
    "mode_share_pct",
    "mode_share_overall_pct",
    "mode_baselines_overall",
    "distribution",
    "numeric_min",
    "numeric_max",
    "status",
]


def eligible_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep all distinct successful finite-scored runs in one launch/cohort.

    Repeated seeds and reruns count separately, intentionally. Missing snapshots
    remain eligible for ranking but contribute no observed parameter values.
    """
    if frame.empty:
        return frame.copy()
    required = {"run_id", "status", "primary_value", *COHORT}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing run columns: {sorted(missing)}")
    # Failed/running rows may not have a primary_metric yet; compare scored rows only.
    cohort_scores = pd.to_numeric(frame["primary_value"], errors="coerce")
    comparable = frame.loc[
        frame["status"].eq("succeeded") & np.isfinite(cohort_scores.fillna(np.nan).astype(float))
    ]
    if not comparable.empty and (
        comparable[COHORT].isna().any().any() or len(comparable[COHORT].drop_duplicates()) != 1
    ):
        raise ValueError(
            "Cannot average the whole launch across different or incomplete dataset/split/metric cohorts"
        )
    if "experiment_run_id" in frame and frame["experiment_run_id"].nunique(dropna=False) > 1:
        raise ValueError("Select one sweep launch")
    if frame["run_id"].isna().any() or frame["run_id"].duplicated().any():
        raise ValueError("Expected one row per non-null run_id")
    scores = pd.to_numeric(frame["primary_value"], errors="coerce")
    mask = frame["status"].eq("succeeded") & np.isfinite(scores.fillna(np.nan).astype(float))
    out = frame.loc[mask].copy().reset_index(drop=True)
    out["primary_value"] = scores.loc[mask].to_numpy(dtype=float)
    if "params_recorded" in out:
        unrecorded = ~out["params_recorded"].eq(True).fillna(False)
        for column in parameter_columns(out):
            out[column] = out[column].astype(object)
            out.loc[unrecorded, column] = None
    return out


def parameter_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame if isinstance(c, str) and c.startswith(PARAM_PREFIX))


def _values(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].dropna().astype(str)


def _ordered(values) -> list[str]:
    values = list(values)
    numbers = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce")
    if len(values) and np.isfinite(numbers.to_numpy(dtype=float)).all():
        return sorted(values, key=lambda value: (float(value), value))
    return sorted(values)


def parameter_summaries(
    frame: pd.DataFrame, *, k: int = 10, higher_is_better: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return overall value-level performance, top-k concentration and counts.

    Overall std uses ddof=1 across runs (not standard error or seed-only spread).
    Top-k ties break by ascending run_id. All modal ties are retained; the
    baseline reports the share of each of those same mode values overall.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    usable = eligible_runs(frame)
    columns = parameter_columns(usable)
    if not usable.empty:
        usable["_run_order"] = usable["run_id"].astype(str)
        usable = usable.sort_values(
            ["primary_value", "_run_order"],
            ascending=[not higher_is_better, True],
            kind="stable",
        ).drop(columns="_run_order")
    top = usable.head(k)
    overall_rows, stability_rows = [], []
    for column in columns:
        parameter = column.removeprefix(PARAM_PREFIX)
        all_values = _values(usable, column)
        all_counts = all_values.value_counts()
        status = "Fixed throughout" if len(all_counts) == 1 else "Varied"
        if all_counts.empty:
            status = "No recorded values"
        rows = []
        for value in _ordered(all_counts.index):
            scores = usable.loc[all_values.index[all_values.eq(value)], "primary_value"]
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "n_runs": len(scores),
                    "mean": float(scores.mean()),
                    "std": float(scores.std(ddof=1)) if len(scores) > 1 else None,
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "status": status,
                }
            )
        rows.sort(
            key=lambda row: (
                -row["mean"] if higher_is_better else row["mean"],
                _ordered(all_counts.index).index(row["value"]),
            )
        )
        overall_rows.extend(rows)
        selected = _values(top, column)
        counts = selected.value_counts()
        n = len(selected)
        modes = _ordered(counts.index[counts.eq(counts.max())]) if n else []
        numeric = pd.to_numeric(selected, errors="coerce")
        is_numeric = n > 0 and np.isfinite(numeric.to_numpy(dtype=float)).all()
        stability_rows.append(
            {
                "parameter": parameter,
                "n_runs": n,
                "missing_runs": len(top) - n,
                "distinct_values": len(counts),
                "mode_values": modes,
                "mode_share_pct": 100.0 * int(counts.max()) / n if n else None,
                "mode_share_overall_pct": (
                    100.0 * int(all_counts[modes[0]]) / len(all_values) if len(modes) == 1 else None
                ),
                "mode_baselines_overall": [
                    {"value": mode, "share_pct": 100.0 * int(all_counts[mode]) / len(all_values)}
                    for mode in modes
                ],
                "distribution": [
                    {"value": value, "count": int(counts[value])}
                    for value in _ordered(counts.index)
                ],
                "numeric_min": float(numeric.min()) if is_numeric else None,
                "numeric_max": float(numeric.max()) if is_numeric else None,
                "status": status,
            }
        )
    stability = pd.DataFrame(stability_rows, columns=STABILITY_COLUMNS)
    if not stability.empty:
        stability = stability.sort_values(
            ["mode_share_pct", "parameter"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    metadata = {
        "registered_runs": len(frame),
        "eligible_runs": len(usable),
        "excluded_runs": len(frame) - len(usable),
        "requested_k": int(k),
        "selected_runs": len(top),
        "parameter_count": len(columns),
        "insufficient_runs": len(top) < 2,
        "model_types": (
            sorted(usable["model_type"].dropna().astype(str).unique().tolist())
            if "model_type" in usable
            else []
        ),
    }
    return pd.DataFrame(overall_rows, columns=OVERALL_COLUMNS), stability, metadata
