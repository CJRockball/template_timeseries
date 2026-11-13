"""Item 9: Drift-to-retrain decision policy engine.

One focused policy evaluation on index_monitor.csv:
Inspects Phase-II alarm rate delta (max_delta) or triage excursion trends
across sequential production monitoring records.

Exit codes for CI / Makefile composition:
  0: IN_CONTROL (no retrain needed)
  1: RETRAIN_RECOMMENDED (alarm delta threshold exceeded)

Entry point: project-retrain-trigger = project.retrain_trigger:main
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ml_template.paths import ARTIFACTS, LOGS
from ml_template.tracking.utils import setup_logging

logger = logging.getLogger(__name__)

INDEX_MONITOR = ARTIFACTS / "index_monitor.csv"
DECISION_DIR = ARTIFACTS / "retrain"


def evaluate_retrain_trigger(
    index_path: Path = INDEX_MONITOR,
    *,
    delta_threshold: float = 0.05,
    consecutive_runs: int = 1,
) -> dict:
    """Evaluate whether historical Phase-II alarm drift exceeds retrain policy.

    Parameters
    ----------
    index_path : Path
        Path to artifacts/index_monitor.csv.
    delta_threshold : float
        Phase-II minus Phase-I alarm rate threshold defining sensor drift (default: 0.05).
    consecutive_runs : int
        Number of consecutive monitoring runs that must breach the threshold
        to trigger a retrain recommendation (default: 1).

    Returns
    -------
    dict
        Structured verdict dictionary with decision details.
    """
    if not index_path.exists():
        return {
            "retrain_recommended": False,
            "verdict": "NO_DATA",
            "reason": f"Ledger file not found: {index_path}",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }

    # Explicit 0-byte file check before pd.read_csv to avoid EmptyDataError
    if index_path.stat().st_size == 0:
        return {
            "retrain_recommended": False,
            "verdict": "EMPTY",
            "reason": f"File is empty: {index_path}",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }

    try:
        df = pd.read_csv(index_path).fillna("")
    except pd.errors.EmptyDataError:
        return {
            "retrain_recommended": False,
            "verdict": "EMPTY",
            "reason": f"No data or columns to parse: {index_path}",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }
    except Exception as e:
        return {
            "retrain_recommended": False,
            "verdict": "ERROR",
            "reason": f"Could not parse ledger {index_path}: {e}",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }

    if df.empty:
        return {
            "retrain_recommended": False,
            "verdict": "EMPTY",
            "reason": f"No records found in {index_path}",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }

    # Focus on records that carry max_delta or drift metrics (spc or monitoring)
    # Check max_delta column if present, or verdict column if produced by monitor.py
    if "max_delta" in df.columns:
        valid_records = df[pd.to_numeric(df["max_delta"], errors="coerce").notna()].copy()
        metric_col = "max_delta"
    elif "drift_fraction" in df.columns:
        valid_records = df[pd.to_numeric(df["drift_fraction"], errors="coerce").notna()].copy()
        metric_col = "drift_fraction"
    else:
        return {
            "retrain_recommended": False,
            "verdict": "NO_METRICS",
            "reason": "Neither max_delta nor drift_fraction found in ledger",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": len(df),
        }

    if valid_records.empty:
        return {
            "retrain_recommended": False,
            "verdict": "NO_VALID_RECORDS",
            "reason": f"No numeric {metric_col} records to inspect",
            "threshold": delta_threshold,
            "consecutive_runs_required": consecutive_runs,
            "records_inspected": 0,
        }

    recent = valid_records.tail(consecutive_runs)
    metric_values = [float(x) for x in recent[metric_col]]
    excursions = [v >= delta_threshold for v in metric_values]

    # If the run explicitly has a 'verdict' column indicating RETRAIN_RECOMMENDED, flag it
    explicit_retrain = (
        "verdict" in recent.columns and recent["verdict"].eq("RETRAIN_RECOMMENDED").any()
    )

    retrain = bool(explicit_retrain or (len(recent) >= consecutive_runs and all(excursions)))

    verdict_str = "RETRAIN_RECOMMENDED" if retrain else "IN_CONTROL"
    reason = (
        f"Observed recent {metric_col} values {metric_values} against threshold {delta_threshold}"
        if not explicit_retrain
        else "Explicit RETRAIN_RECOMMENDED verdict found in recent monitoring batch"
    )

    return {
        "retrain_recommended": retrain,
        "verdict": verdict_str,
        "reason": reason,
        "metric_inspected": metric_col,
        "recent_values": metric_values,
        "threshold": delta_threshold,
        "consecutive_runs_required": consecutive_runs,
        "records_inspected": len(recent),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Item 9: Evaluate Phase-II alarm drift from index_monitor.csv -> retrain recommendation"
    )
    p.add_argument(
        "--index",
        default=INDEX_MONITOR,
        type=Path,
        help=f"Path to index_monitor.csv (default: {INDEX_MONITOR})",
    )
    p.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Alarm delta threshold above which drift triggers a retrain recommendation (default: 0.05)",
    )
    p.add_argument(
        "--consecutive-runs",
        type=int,
        default=1,
        help="Number of consecutive breached runs required (default: 1)",
    )
    p.add_argument(
        "--json-out",
        action="store_true",
        help="Print decision output as structured JSON to stdout",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "retrain_trigger.log")

    decision = evaluate_retrain_trigger(
        index_path=args.index,
        delta_threshold=args.delta_threshold,
        consecutive_runs=args.consecutive_runs,
    )

    if args.json_out:
        print(json.dumps(decision, indent=2))
    else:
        logger.info("[retrain_trigger] Verdict: %s", decision["verdict"])
        logger.info(
            "[retrain_trigger] Recommended: %s | %s",
            decision["retrain_recommended"],
            decision["reason"],
        )

    # Persist decision artifact for auditability
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = DECISION_DIR / "latest_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Exit code: 0 = in control, 1 = retrain recommended
    if decision["retrain_recommended"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
