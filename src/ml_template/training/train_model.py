"""Train one config or a sweep.

    python -m ml_template.training.train_model --config configs/experiment/xgb_baseline.yaml
    python -m ml_template.training.train_model --config <file> --set model.params.max_depth=6
    python -m ml_template.training.train_model --sweep configs/sweep/xgb_grid.yaml

Needs Silver and the split step first. Exit code 1 if any run failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ml_template.db.connection import get_engine
from ml_template.tracking.pipeline_logging import setup_logging
from ml_template.training.experiment_runner import run_experiment
from ml_template.training.run_config import expand_sweep, load_config

logger = logging.getLogger("ml_template.training.train_model")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a model from a run config or a sweep.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--sweep", type=Path)
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    p.add_argument("--fail-fast", action="store_true", help="stop a sweep at the first failure")
    args = p.parse_args(argv)
    if args.sweep and args.overrides:
        p.error("--set only works with --config; put overrides in the sweep file")

    setup_logging()
    if args.sweep:
        configs = expand_sweep(args.sweep)
        kind, sweep_path = "sweep", str(args.sweep)
    else:
        configs = [load_config(args.config, args.overrides)]
        kind, sweep_path = "single", None

    summary = run_experiment(
        configs, get_engine(), kind=kind, sweep_path=sweep_path, fail_fast=args.fail_fast
    )
    for failure in summary["failures"]:
        logger.error("failed config %s: %s", failure["config_hash"], failure["error"])
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
