"""Explicit development-only policy selection; never runs a holdout evaluation.

The selected optimum is descriptive on the rows used for selection. Calibrated
cross-fit cutoffs require explicit acknowledgement of transfer to one final mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_template.evaluation.threshold import (
    budget_boundaries,
    operating_point,
    recommendations,
    threshold_profile,
)

DATA_FIELDS = (
    "silver_dataset_version_id",
    "feature_set_id",
    "split_id",
    "silver_parquet_sha256",
    "silver_manifest_sha256",
    "split_manifest_sha256",
    "split_assignments_sha256",
)
SOURCE_FILES = ("development_predictions.parquet", "calibrator.json", "provenance.json")
DISCLAIMER = (
    "Selected and described on development OOF score-label pairs; maximised metrics "
    "are selection-biased, not independent validation. A pooled calibrated-cross-fit "
    "cutoff transfers approximately, not exactly, to one final calibrator."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def selection_candidate(y, p, objective, value=None):
    """Pure candidate generation. No fitting and no assertion of unbiased performance."""
    if objective in {"f1", "f2"}:
        if value is not None:
            raise ValueError("do not supply a manual value for an objective search")
        table = recommendations(threshold_profile(y, p))
        choice = table.loc[table.objective.eq(objective)].iloc[0].to_dict()
        threshold = choice.get("threshold")
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                "no deployable defined F-score optimum; both policy and label support must be reviewed"
            )
        return {
            "kind": "threshold",
            "value": float(threshold),
            "objective": objective,
            "diagnostic": json_safe(choice),
        }
    if objective == "manual":
        if value is None or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("manual objective requires --value in [0,1]")
        return {
            "kind": "threshold",
            "value": float(value),
            "objective": objective,
            "diagnostic": json_safe(operating_point(y, p, value)),
        }
    if objective == "alert_rate":
        if value is None:
            raise ValueError("alert_rate objective requires --value")
        row = budget_boundaries(y, p, rates=(value,)).iloc[0].to_dict()
        return {
            "kind": "alert_rate",
            "value": float(value),
            "objective": objective,
            "diagnostic": json_safe(row),
        }
    raise ValueError("objective must be f1, f2, manual or alert_rate")


def load_selection_inputs(engine, parent_run_id, calibration_id):
    from ml_template.db import calibration_registry
    from ml_template.models.calibration_artifacts import load_calibrator_for_run
    from ml_template.tracking import run_artifacts as artifacts
    from ml_template.training.calibration_inputs import load_calibration_inputs

    load_calibrator_for_run(engine, calibration_id, parent_run_id)
    bound = load_calibration_inputs(engine, parent_run_id)
    record = calibration_registry.get_calibration(engine, calibration_id)
    folder = artifacts.resolve_stored_path(record["artifact_dir"])
    for name in SOURCE_FILES:
        artifacts.verify_artifact(folder, name)
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    for name in DATA_FIELDS:
        if provenance[name] != bound.provenance[name]:
            raise ValueError(f"calibration data provenance mismatch: {name}")
    key = bound.key_col
    columns = [key, "fold", "y_true", "p_raw", "p_cal_crossfit"]
    frame = pd.read_parquet(folder / "development_predictions.parquet", columns=columns)
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError("development prediction keys must be unique and non-null")
    base = bound.frame.set_index(key).sort_index()
    frame = frame.set_index(key).sort_index()
    if not frame.index.equals(base.index):
        raise ValueError("prediction keys differ from the exact stored development population")
    for name in ["fold", "y_true", "p_raw"]:
        if not np.array_equal(frame[name].to_numpy(), base[name].to_numpy()):
            raise ValueError(f"stored development predictions disagree with parent: {name}")
    for name in ["p_raw", "p_cal_crossfit"]:
        p = frame[name].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("invalid diagnostic probabilities")
    return {
        "frame": frame.reset_index(),
        "key_col": key,
        "provenance": provenance,
        "source_hashes": {n: sha256(folder / n) for n in SOURCE_FILES},
    }


def validate_selection_record(spec, record, provenance, source_hashes):
    """Validate an already chosen policy. Never search for a new cutoff here."""
    ref = spec.policy.selection
    if ref is None:
        return None
    if record.get("schema_version") != 1:
        raise ValueError("unsupported policy-selection record")
    for field in [
        "parent_run_id",
        "calibration_id",
        "score_variant",
        "objective",
        "transfer_acknowledged",
    ]:
        if record.get(field) != getattr(ref, field):
            raise ValueError(f"policy selection reference mismatch: {field}")
    if record.get("batch_definition") != spec.policy.batch_definition:
        raise ValueError("frozen policy batch definition differs from selection")
    if record.get("kind") != spec.policy.kind or record.get("value") != spec.policy.value:
        raise ValueError("frozen policy differs from selected policy")
    if (
        record.get("parent_run_id") != spec.parent_run_id
        or record.get("calibration_id") != spec.calibration_id
    ):
        raise ValueError("selected policy belongs to another predictor")
    expected_variant = "calibrated_crossfit" if spec.release_variant == "calibrated" else "raw"
    if record.get("score_variant") != expected_variant:
        raise ValueError("policy score scale does not match release variant")
    if expected_variant == "calibrated_crossfit" and not ref.transfer_acknowledged:
        raise ValueError("cross-fit-to-final calibrator transfer must be acknowledged")
    for name in DATA_FIELDS:
        if record.get("data_identity", {}).get(name) != provenance.get(name):
            raise ValueError(f"selected policy data identity changed: {name}")
    if record.get("parent_files_sha256") != provenance.get("parent_files_sha256"):
        raise ValueError("selected policy parent model/config/OOF artifacts changed")
    if set(record.get("source_hashes", {})) != set(SOURCE_FILES):
        raise ValueError("incomplete policy source hashes")
    if record["source_hashes"] != source_hashes:
        raise ValueError("policy source prediction/calibrator artifacts changed")
    return record


def validate_policy_selection(spec, provenance, calibration_dir, *, repo_root=None):
    """Preflight verification, before any holdout-label access."""
    ref = spec.policy.selection
    if ref is None:
        return None
    from ml_template.paths import ROOT
    from ml_template.tracking import run_artifacts as artifacts

    path = Path(ref.artifact_path)
    if not path.is_absolute():
        path = Path(repo_root or ROOT) / path
    if sha256(path) != ref.artifact_sha256:
        raise ValueError("policy-selection record hash mismatch")
    if calibration_dir is None:
        raise ValueError("this selection record requires its source calibration")
    hashes = {}
    for name in SOURCE_FILES:
        artifacts.verify_artifact(Path(calibration_dir), name)
        hashes[name] = sha256(Path(calibration_dir) / name)
    record = json.loads(path.read_text(encoding="utf-8"))
    return validate_selection_record(spec, record, provenance, hashes)


def save_selection(
    inputs,
    *,
    parent_run_id,
    calibration_id,
    score_variant,
    objective,
    value=None,
    acknowledge_transfer=False,
    output_root,
    evaluation_template=None,
    batch_definition="entire scoring batch",
):
    from ml_template.evaluation.evaluation_config import EvaluationConfig

    if score_variant not in {"raw", "calibrated_crossfit"}:
        raise ValueError("only raw or calibrated_crossfit diagnostics are supported")
    if score_variant == "calibrated_crossfit" and not acknowledge_transfer:
        raise ValueError(
            "explicit --acknowledge-transfer required for cross-fit-to-final mapping transfer"
        )
    frame = inputs["frame"].sort_values(inputs["key_col"], kind="stable")
    if inputs["provenance"].get("parent_run_id") != parent_run_id:
        raise ValueError("selection inputs belong to another parent")
    column = "p_raw" if score_variant == "raw" else "p_cal_crossfit"
    candidate = selection_candidate(frame.y_true, frame[column], objective, value)
    sid = f"policy_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    provenance = inputs["provenance"]
    record = {
        "schema_version": 1,
        "selection_id": sid,
        "created_at": datetime.now(UTC).isoformat(),
        "parent_run_id": parent_run_id,
        "calibration_id": calibration_id,
        "score_variant": score_variant,
        "transfer_acknowledged": acknowledge_transfer,
        **candidate,
        "batch_definition": batch_definition,
        "n_rows": len(frame),
        "n_positives": int(frame.y_true.sum()),
        "data_identity": {k: provenance[k] for k in DATA_FIELDS},
        "parent_files_sha256": provenance["parent_files_sha256"],
        "source_hashes": inputs["source_hashes"],
        "threshold_rule": "p >= cutoff",
        "tie_break": "highest cutoff among tied objective maxima; budget ties use ascending entity key",
        "assessment": DISCLAIMER,
    }
    # Validate template before writing any output. It is a draft, not a holdout execution.
    base = evaluation_template or EvaluationConfig(parent_run_id=parent_run_id)
    template = base.model_dump(mode="json", exclude_none=True)
    out = (Path(output_root) / sid).resolve()
    selection_path = out / "selection.json"
    encoded = json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    reference = {
        "artifact_path": str(selection_path),
        "artifact_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "parent_run_id": parent_run_id,
        "calibration_id": calibration_id,
        "score_variant": score_variant,
        "objective": objective,
        "transfer_acknowledged": acknowledge_transfer,
    }
    if evaluation_template and (
        base.parent_run_id != parent_run_id or base.calibration_id != calibration_id
    ):
        raise ValueError("evaluation template must already identify this parent and calibration")
    template.update(
        parent_run_id=parent_run_id,
        calibration_id=calibration_id,
        release_variant="calibrated" if score_variant == "calibrated_crossfit" else "raw",
        policy={
            "kind": candidate["kind"],
            "value": candidate["value"],
            "source": f"Explicit development selection {sid}: {objective}",
            "batch_definition": batch_definition,
            "selection": reference,
        },
    )
    limitations = list(template.get("limitations", []))
    if DISCLAIMER not in limitations:
        limitations.append(DISCLAIMER)
    template["limitations"] = limitations
    cfg = EvaluationConfig.model_validate(template)
    out.mkdir(parents=True, exist_ok=False)
    selection_path.write_text(encoded, encoding="utf-8")
    (out / "evaluation_draft.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    (out / "README.txt").write_text(
        DISCLAIMER
        + "\nReview intended use, criteria, limitations and batch definition before evaluation. "
        "No holdout evaluation was executed. Preserve selection.json unchanged.\n",
        encoding="utf-8",
    )
    return {
        "selection_id": sid,
        "selection_file": str(selection_path),
        "evaluation_draft": str(out / "evaluation_draft.yaml"),
        "policy": candidate,
        "holdout_labels_loaded": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--score-variant", choices=["raw", "calibrated_crossfit"], required=True)
    parser.add_argument("--objective", choices=["f1", "f2", "manual", "alert_rate"], required=True)
    parser.add_argument("--value", type=float)
    parser.add_argument("--acknowledge-transfer", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evaluation-template", type=Path)
    parser.add_argument("--batch-definition", default="entire scoring batch")
    parser.add_argument(
        "--check", action="store_true", help="inspect proposed policy without writing"
    )
    args = parser.parse_args(argv)
    from ml_template.db.connection import get_engine
    from ml_template.evaluation.evaluation_config import load_evaluation_config
    from ml_template.paths import ARTIFACTS

    inputs = load_selection_inputs(get_engine(), args.parent_run_id, args.calibration_id)
    if args.check:
        column = "p_raw" if args.score_variant == "raw" else "p_cal_crossfit"
        result = {
            "parent_run_id": args.parent_run_id,
            "calibration_id": args.calibration_id,
            "score_variant": args.score_variant,
            "candidate": selection_candidate(
                inputs["frame"].y_true, inputs["frame"][column], args.objective, args.value
            ),
            "transfer_acknowledgement_required": args.score_variant == "calibrated_crossfit",
            "assessment": DISCLAIMER,
            "files_written": False,
            "holdout_labels_loaded": False,
        }
    else:
        result = save_selection(
            inputs,
            parent_run_id=args.parent_run_id,
            calibration_id=args.calibration_id,
            score_variant=args.score_variant,
            objective=args.objective,
            value=args.value,
            acknowledge_transfer=args.acknowledge_transfer,
            output_root=args.output_root or ARTIFACTS / "policies",
            evaluation_template=load_evaluation_config(args.evaluation_template)
            if args.evaluation_template
            else None,
            batch_definition=args.batch_definition,
        )
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
