"""Load an immutable calibrator only with its explicitly bound parent run."""

from __future__ import annotations

import json

from ml_template.models.probability_calibrator import ProbabilityCalibrator


def load_calibrator_for_run(engine, calibration_id: str, parent_run_id: str):
    from ml_template.db import calibration_registry, model_run_registry
    from ml_template.tracking import run_artifacts

    record = calibration_registry.get_calibration(engine, calibration_id)
    if not record or record["status"] != "succeeded":
        raise ValueError("calibration must exist and have succeeded")
    if record["parent_run_id"] != parent_run_id:
        raise ValueError("calibration belongs to a different parent run")
    parent = model_run_registry.get_run(engine, parent_run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("parent run must exist and have succeeded")
    out = run_artifacts.resolve_stored_path(record["artifact_dir"])
    for name in ["calibrator.json", "provenance.json", "config.json"]:
        run_artifacts.verify_artifact(out, name)
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if provenance["parent_run_id"] != parent_run_id or config["parent_run_id"] != parent_run_id:
        raise ValueError("calibration parent metadata mismatch")
    parent_dir = run_artifacts.resolve_stored_path(parent["artifact_dir"])
    for name, expected in provenance["parent_files_sha256"].items():
        run_artifacts.verify_artifact(parent_dir, name)
        if run_artifacts.sha256_file(parent_dir / name) != expected:
            raise ValueError(f"parent artifact changed since calibration: {name}")
    cal = ProbabilityCalibrator.load(out / "calibrator.json")
    if cal.method != config["method"]:
        raise ValueError("calibration method metadata mismatch")
    return cal
