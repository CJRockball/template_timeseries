"""Score one explicit release against an identified, canonical, unlabelled batch."""

import argparse
import json
import logging
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_template.scoring.input_contract import validate_batch
from ml_template.scoring.predictor import ReleasedPredictor
from ml_template.scoring.release_artifacts import file_entries, sha256, write_json

logger = logging.getLogger(__name__)


def read_batch(path, contract):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        kwargs = {"dtype": {contract["key_col"]: "string"}} if contract["key_kind"] == "str" else {}
        frame = pd.read_csv(path, **kwargs)
    else:
        raise ValueError("input must be canonical .csv or .parquet")
    validate_batch(frame, contract, enforce_rates=True)
    return frame


def check_batch_definition(predictor, batch_definition):
    if predictor.policy.kind == "alert_rate" and batch_definition is None:
        raise ValueError("alert-rate scoring requires explicit batch-definition acknowledgement")
    if batch_definition is not None and batch_definition != predictor.policy.batch_definition:
        raise ValueError("batch definition differs from the frozen release policy")


def _score_batch(
    engine,
    release_dir,
    input_path,
    *,
    output_root=None,
    chunk_size=None,
    batch_definition=None,
    check_only=False,
    batch_metadata=None,
    stage_hook=None,
):
    stage_hook = stage_hook or (lambda stage, batch_id=None: None)
    stage_hook("release_loading")
    predictor = ReleasedPredictor(release_dir)
    stage_hook("argument_validation")
    check_batch_definition(predictor, batch_definition)
    input_path = Path(input_path).resolve()
    if input_path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("input must be .csv or .parquet")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if check_only:
        before = sha256(input_path)
        frame = read_batch(input_path, predictor.contract)
        if (getattr(predictor, "info", None) or {}).get("task") == "regression":
            predictor.prepare_features(frame)
        if sha256(input_path) != before:
            raise ValueError("input changed during preflight")
        return {
            "ready": True,
            "release_id": predictor.release_id,
            "n_rows": len(frame),
            "input_sha256": before,
            "files_written": False,
            "predictions_computed": False,
            **(
                {"stateless_recipe_checked": True}
                if (getattr(predictor, "info", None) or {}).get("task") == "regression"
                else {}
            ),
        }
    from ml_template.db import scoring_registry as registry
    from ml_template.paths import ARTIFACTS, LOGS
    from ml_template.tracking.pipeline_logging import run_log

    batch_id = f"score_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory = (Path(output_root or ARTIFACTS / "scores") / batch_id).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage_hook("registration")
    registry.register_release(engine, predictor.manifest, predictor.info, predictor.directory)
    registry.start_batch(engine, batch_id, predictor.release_id, directory, input_path.name)
    stage_hook("input_snapshot", batch_id)
    started = time.perf_counter()
    digest = None
    try:
        with run_log(LOGS / "scoring" / f"{batch_id}.log"):
            logger.info("Scoring %s started: release=%s", batch_id, predictor.release_id)
            directory.mkdir(exist_ok=False)
            snapshot = directory / f"input{input_path.suffix.lower()}"
            before = sha256(input_path)
            shutil.copyfile(input_path, snapshot)
            digest = sha256(snapshot)
            if before != digest or digest != sha256(input_path):
                raise ValueError("input changed while snapshotting")
            stage_hook("input_validation", batch_id)
            frame = read_batch(snapshot, predictor.contract)
            stage_hook("prediction", batch_id)
            prediction_started = time.perf_counter()
            out = predictor.predict(frame, chunk_size=chunk_size)
            prediction_duration = time.perf_counter() - prediction_started
            stage_hook("artifact_writing", batch_id)
            out["scoring_batch_id"] = batch_id
            out.to_parquet(directory / "predictions.parquet", index=False)
            flagged = int(out.policy_flag.sum()) if "policy_flag" in out else None
            summary = {
                "schema_version": 1,
                "scoring_batch_id": batch_id,
                "release_id": predictor.release_id,
                "content_sha256": predictor.manifest["content_sha256"],
                "parent_run_id": predictor.info["parent_run_id"],
                "calibration_id": predictor.info["calibration_id"],
                "policy": predictor.info["policy"],
                "batch_definition": predictor.policy.batch_definition,
                "batch_definition_acknowledged": batch_definition is not None,
                "input_name": input_path.name,
                "input_sha256": digest,
                "n_rows": len(out),
                "n_flagged": flagged,
                "flagged_fraction": flagged / len(out) if flagged is not None else None,
                "release_variant": predictor.info["release_variant"],
                "chunk_size": chunk_size,
                "labels_loaded": False,
                "created_at": datetime.now(UTC).isoformat(),
                "duration_s": time.perf_counter() - started,
                "batch_metadata": batch_metadata or {},
                "timings": {"prediction_s": prediction_duration},
            }
            if predictor.info.get("task") == "regression":
                summary.update(
                    schema_version=2,
                    task="regression",
                    prediction_kind="continuous",
                    prediction_columns=predictor.info["prediction_columns"],
                )
            if predictor.info.get("postprocessing") is not None:
                summary.update(
                    target_unit=predictor.info["target_unit"],
                    interval_columns=predictor.info["interval_columns"],
                    postprocessing=predictor.info["postprocessing"],
                )
            write_json(directory / "summary.json", summary)
            write_json(
                directory / "artifacts.json",
                {"files": file_entries(directory, exclude=("artifacts.json",))},
            )
            logger.info("Scoring %s completed: rows=%d flagged=%s", batch_id, len(out), flagged)
    except Exception as exc:
        registry.finish_batch(
            engine,
            batch_id,
            duration_s=time.perf_counter() - started,
            input_sha256=digest,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    stage_hook("batch_finalization", batch_id)
    registry.finish_batch(
        engine,
        batch_id,
        duration_s=time.perf_counter() - started,
        input_sha256=digest,
        n_rows=len(out),
        n_flagged=flagged,
    )
    return {
        "scoring_batch_id": batch_id,
        "release_id": predictor.release_id,
        "status": "succeeded",
        "artifact_dir": str(directory),
        "n_rows": len(out),
        "n_flagged": flagged,
    }


def score_batch(
    engine,
    release_dir,
    input_path,
    *,
    output_root=None,
    chunk_size=None,
    batch_definition=None,
    check_only=False,
    batch_metadata=None,
):
    from ml_template.monitoring.scoring_attempts import tracked_score

    return tracked_score(
        _score_batch,
        engine,
        release_dir,
        input_path,
        output_root=output_root,
        chunk_size=chunk_size,
        batch_definition=batch_definition,
        check_only=check_only,
        batch_metadata=batch_metadata,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-definition")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--metadata", type=Path, help="Optional batch metadata JSON")
    args = parser.parse_args(argv)
    from ml_template.tracking.pipeline_logging import setup_logging

    setup_logging()
    engine = None
    if not args.check:
        from ml_template.db.connection import get_engine

        engine = get_engine()
    result = score_batch(
        engine,
        args.release,
        args.input,
        output_root=args.output_root,
        chunk_size=args.chunk_size,
        batch_definition=args.batch_definition,
        check_only=args.check,
        batch_metadata=args.metadata,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
