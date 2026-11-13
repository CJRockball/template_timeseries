"""Post-selection OOF calibration. Never loads holdout predictions or labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.evaluation.calibration_diagnostics import (
    cross_fitted_probabilities,
    probability_metrics,
    reliability_table,
)
from ml_template.models.probability_calibrator import ProbabilityCalibrator
from ml_template.training.calibration_config import load_calibration_config
from ml_template.training.calibration_inputs import load_calibration_inputs

logger = logging.getLogger(__name__)


def calibrate_run(engine, cfg, *, output_root=None, check_only=False):
    from ml_template.db import calibration_registry, model_run_registry
    from ml_template.paths import ARTIFACTS
    from ml_template.tracking import run_artifacts
    from ml_template.tracking.git_state import git_state

    inputs = load_calibration_inputs(engine, cfg.parent_run_id)
    frame = inputs.frame
    counts = np.bincount(frame["y_true"].to_numpy(dtype=int), minlength=2)
    if counts.min() < cfg.min_class_count:
        raise ValueError(f"insufficient class counts for calibration: {counts.tolist()}")
    if check_only:
        return {
            "parent_run_id": cfg.parent_run_id,
            "n_rows": len(frame),
            "class_counts": counts.tolist(),
            "provenance": inputs.provenance,
        }
    if inputs.provenance["parent_holdout_evaluated_at"]:
        logger.warning(
            "Parent holdout was already evaluated; this cannot undo earlier selection exposure"
        )
    calibration_id = model_run_registry.new_run_id("cal")
    out = Path(output_root or ARTIFACTS / "calibration") / calibration_id
    out.mkdir(parents=True, exist_ok=False)
    calibration_registry.start_calibration(
        engine,
        calibration_id,
        cfg.parent_run_id,
        cfg.method,
        cfg.model_dump(mode="json"),
        run_artifacts.relative_to_root(out),
    )
    try:
        settings = {"method": cfg.method, "epsilon": cfg.epsilon, "l2": cfg.l2}
        p_raw = frame["p_raw"].to_numpy()
        y = frame["y_true"].to_numpy()
        p_crossfit = cross_fitted_probabilities(
            p_raw, y, frame["fold"].to_numpy(), min_class_count=cfg.min_class_count, **settings
        )
        calibrator = ProbabilityCalibrator.fit(p_raw, y, **settings)
        calibrator.save(out / "calibrator.json")
        predictions = frame.copy()
        predictions["p_cal_crossfit"] = p_crossfit
        predictions["p_cal_fit"] = calibrator.transform(p_raw)
        predictions.to_parquet(out / "development_predictions.parquet", index=False)
        rows = []
        reliability = []
        masks = [(-1, np.ones(len(frame), dtype=bool))]
        masks += [(int(k), frame["fold"].eq(k).to_numpy()) for k in sorted(frame["fold"].unique())]
        for variant, p in [("raw", p_raw), ("calibrated_crossfit", p_crossfit)]:
            for fold, mask in masks:
                for metric, value in probability_metrics(y[mask], p[mask]).items():
                    rows.append(
                        {
                            "split": "dev_calibration_diagnostic",
                            "variant": variant,
                            "fold": fold,
                            "metric": metric,
                            "value": value,
                        }
                    )
            table = reliability_table(y, p, cfg.reliability_bins)
            table.insert(0, "variant", variant)
            reliability.append(table)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        pd.concat(reliability, ignore_index=True).to_csv(out / "reliability.csv", index=False)
        run_artifacts.write_json(out / "config.json", cfg.model_dump(mode="json"))
        run_artifacts.write_json(out / "provenance.json", inputs.provenance)
        run_artifacts.write_json(out / "environment.json", run_artifacts.environment_info())
        run_artifacts.write_json(out / "git_state.json", git_state())
        run_artifacts.write_json(
            out / "diagnostic_notes.json",
            {
                "assessment": "exploratory cross-fitting of stored OOF score-label pairs; not nested CV",
                "p_cal_fit": "fitted on all OOF pairs; in-sample for calibrator; not assessment evidence",
                "p_cal_crossfit": "each row excluded from its calibrator fit; not independent of base-model selection",
                "method": "positive-slope sigmoid on clipped probability logit; weak L2 penalty on slope",
                "ranking": "final calibrator is nondecreasing; clipping/numerics may create ties; fold-specific diagnostic maps can reorder pooled rows",
                "reliability_intervals": "pointwise Wilson intervals assuming independent Bernoulli rows; exploratory, not simultaneous",
                "holdout": "no holdout predictions or labels loaded; file hashing reads bytes for integrity only",
                "warning_slope_at_floor": calibrator.slope <= 1.1e-8,
            },
        )
        run_artifacts.write_artifact_manifest(out)
        calibration_registry.finish_calibration(engine, calibration_id, succeeded=True, rows=rows)
    except Exception as exc:
        calibration_registry.finish_calibration(
            engine, calibration_id, succeeded=False, error=f"{type(exc).__name__}: {exc}"[:2000]
        )
        raise
    return {
        "calibration_id": calibration_id,
        "parent_run_id": cfg.parent_run_id,
        "artifact_dir": str(out),
        "status": "succeeded",
    }


def main(argv=None):
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate parent inputs without fitting or new calibration records",
    )
    args = parser.parse_args(argv)
    setup_logging()
    result = calibrate_run(
        get_engine(), load_calibration_config(args.config), check_only=args.check
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
