"""Orchestrate one saved run through final evaluation, release and batch scoring.

No training, calibration fitting, ensemble construction, export or Kaggle submission.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def _binary_services():
    from ml_template.db import evaluation_registry, model_run_registry
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.evaluation_config import EvaluationConfig, load_evaluation_config
    from ml_template.scoring.batch import score_batch
    from ml_template.scoring.prepare_input import main as prepare_input
    from ml_template.scoring.release_artifacts import (
        read_json,
        sha256,
        verify,
        verify_runtime,
        write_json,
    )
    from ml_template.scoring.release_builder import build_release
    from ml_template.scoring.release_config import ReleaseConfig

    return SimpleNamespace(
        get_run=model_run_registry.get_run,
        get_evaluation=evaluation_registry.get_evaluation,
        evaluate=evaluate_spec,
        EvaluationConfig=EvaluationConfig,
        load_evaluation_config=load_evaluation_config,
        build_release=build_release,
        ReleaseConfig=ReleaseConfig,
        prepare_input=prepare_input,
        score=score_batch,
        read_json=read_json,
        sha256=sha256,
        verify=verify,
        verify_runtime=verify_runtime,
        write_json=write_json,
    )


def _binary_evaluation_spec(
    services,
    engine,
    run_id,
    *,
    evaluation_id=None,
    evaluation_config=None,
    primary_metric=None,
    acknowledge_prior_exposure=False,
    prior_exposure_note="",
    no_uncertainty=False,
):
    if evaluation_id and evaluation_config:
        raise ValueError("supply evaluation_id OR evaluation_config, not both")
    if evaluation_id:
        if primary_metric or acknowledge_prior_exposure or prior_exposure_note or no_uncertainty:
            raise ValueError(
                "an existing evaluation ID freezes its spec; omit evaluation overrides"
            )
        record = services.get_evaluation(engine, evaluation_id)
        if not record or record["status"] != "succeeded" or record["parent_run_id"] != run_id:
            raise ValueError("evaluation ID must identify a successful evaluation of this run")
        return services.EvaluationConfig.model_validate_json(record["spec_json"])
    if evaluation_config:
        spec = services.load_evaluation_config(evaluation_config)
        if spec.parent_run_id != run_id:
            raise ValueError("evaluation config parent_run_id differs from --run-id")
        if primary_metric or no_uncertainty or acknowledge_prior_exposure or prior_exposure_note:
            raise ValueError("put evaluation settings in the supplied evaluation YAML")
        return spec
    payload = {"parent_run_id": run_id}
    if primary_metric:
        payload["primary_metric"] = primary_metric
    if no_uncertainty:
        payload["uncertainty"] = {"enabled": False}
    if acknowledge_prior_exposure or prior_exposure_note:
        if not acknowledge_prior_exposure or not prior_exposure_note.strip():
            raise ValueError("prior exposure requires both explicit acknowledgement and a note")
        payload.update(acknowledge_prior_exposure=True, prior_exposure_note=prior_exposure_note)
    return services.EvaluationConfig.model_validate(payload)


def _binary_run_pipeline(
    engine,
    *,
    run_id,
    input_path,
    evaluation_spec,
    expected_evaluation_id=None,
    release_dir=None,
    input_format="raw",
    output_root=None,
    chunk_size=None,
    batch_definition=None,
    check_only=True,
    prep_dir=None,
    acknowledge_research_release=False,
    services=None,
):
    """Return evaluation/release/scoring IDs and paths. Each execution has a fresh folder.

    `check_only` checks evaluation eligibility. Without an existing release, downstream
    input/schema and prediction-parity checks are explicitly deferred, not claimed ready.
    Use release_dir to reuse a frozen release; otherwise build a new immutable package.
    """
    s = services or _services()
    source = Path(input_path).resolve()
    if not source.is_file() or source.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("input must be an existing CSV or Parquet file")
    if input_format not in {"raw", "canonical"}:
        raise ValueError("input_format must be raw or canonical")
    if chunk_size is not None and (isinstance(chunk_size, bool) or chunk_size < 1):
        raise ValueError("chunk_size must be positive")
    parent = s.get_run(engine, run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("run_id must identify a successful saved model")
    if evaluation_spec.parent_run_id != run_id:
        raise ValueError("evaluation parent does not match run_id")
    policy = evaluation_spec.policy
    if policy.kind == "alert_rate" and batch_definition is None:
        raise ValueError("alert-rate policy requires an explicit batch_definition")
    if batch_definition is not None and batch_definition != policy.batch_definition:
        raise ValueError("batch_definition differs from the evaluation policy")
    if not check_only and (not acknowledge_research_release):
        raise ValueError("execution requires --acknowledge-research-release")
    before = s.sha256(source)
    preflight = s.evaluate(engine, evaluation_spec, check_only=True)
    cached = preflight.get("status") == "succeeded" and bool(preflight.get("cached"))
    ready = cached or bool(preflight.get("ready"))
    if expected_evaluation_id and preflight.get("evaluation_id") != expected_evaluation_id:
        raise ValueError("requested evaluation no longer matches current predictor/data identity")
    report = {
        "schema_version": 1,
        "parent_run_id": run_id,
        "input_path": str(source),
        "input_sha256": before,
        "input_format": input_format,
        "evaluation_ready": ready,
        "evaluation_preflight": preflight,
        "downstream_checks": "deferred_until_release_exists",
        "predictions_computed": False,
    }
    existing = None
    if release_dir is not None:
        existing = Path(release_dir).resolve()
        manifest = s.verify(existing)
        s.verify_runtime(s.read_json(existing / "runtime.json"))
        info = s.read_json(existing / "release.json")
        if not cached or info["evaluation_id"] != preflight.get("evaluation_id"):
            raise ValueError("existing release must match the successful cached evaluation")
        if (
            info["parent_run_id"] != run_id
            or info["calibration_id"] != evaluation_spec.calibration_id
            or info["release_variant"] != evaluation_spec.release_variant
            or (info["policy"] != evaluation_spec.policy.model_dump(mode="json"))
        ):
            raise ValueError("existing release differs from the requested predictor/policy")
        report.update(release_id=manifest["release_id"], release_dir=str(existing))
        _check_input(s, engine, existing, source, input_format, chunk_size, batch_definition)
        report["downstream_checks"] = "input_checked; execution_and_prediction_parity_not_run"
    if s.sha256(source) != before:
        raise ValueError("input changed during preflight")
    if check_only:
        return {**report, "status": "checked", "ready": ready, "pipeline_artifacts_written": False}
    if not ready:
        raise ValueError("evaluation preflight blocked; review prior exposure and resolved spec")
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "pipelines"
    if prep_dir is not None:
        if input_format != "raw":
            raise ValueError("--prep-dir is only used with raw input")
        prep_dir = Path(prep_dir).expanduser().resolve()
        if prep_dir.exists():
            raise FileExistsError(f"Prepared-input directory already exists: {prep_dir}")
    identity = (
        f"pipe_{datetime.now(ZoneInfo('Asia/Singapore')):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    directory = Path(output_root).resolve() / identity
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        **report,
        "pipeline_id": identity,
        "artifact_dir": str(directory),
        "status": "running",
        "stage": "evaluation",
        "evaluation_spec": evaluation_spec.model_dump(mode="json"),
    }

    def checkpoint():
        temporary = directory / "pipeline.json.tmp"
        s.write_json(temporary, state)
        temporary.replace(directory / "pipeline.json")

    checkpoint()
    try:
        evaluated = preflight if cached else s.evaluate(engine, evaluation_spec, check_only=False)
        if evaluated.get("status") != "succeeded":
            raise ValueError("evaluation did not succeed")
        eid = evaluated["evaluation_id"]
        state.update(evaluation_id=eid, evaluation_dir=evaluated["artifact_dir"], stage="release")
        checkpoint()
        if existing is None:
            request = s.ReleaseConfig(
                name=f"pipeline_{identity}",
                evaluation_id=eid,
                parent_run_id=run_id,
                calibration_id=evaluation_spec.calibration_id,
                intended_use=evaluation_spec.intended_use,
                acknowledge_research_release=True,
                parity_atol=1e-06 if parent.get("model_type") == "nn" else 1e-07,
            )
            s.build_release(engine, request, check_only=True)
            released = s.build_release(engine, request, check_only=False)
            if released.get("status") != "succeeded":
                raise ValueError("release creation did not succeed")
            existing = Path(released["artifact_dir"])
            state.update(release_id=released["release_id"], release_dir=str(existing))
        state["stage"] = "input_preparation"
        checkpoint()
        if s.sha256(source) != before:
            raise ValueError("input changed after evaluation/release")
        _check_input(s, engine, existing, source, input_format, chunk_size, batch_definition)
        canonical = source
        if input_format == "raw":
            prepared = prep_dir if prep_dir is not None else directory / "input"
            with contextlib.redirect_stdout(io.StringIO()):
                s.prepare_input(
                    [
                        "--release",
                        str(existing),
                        "--input",
                        str(source),
                        "--output-dir",
                        str(prepared),
                        "--execute",
                    ]
                )
            canonical = prepared / "data.parquet"
            preparation = s.read_json(prepared / "manifest.json")
            if preparation["input_sha256"] != before:
                raise ValueError("prepared snapshot differs from the requested input")
        if s.sha256(source) != before:
            raise ValueError("input changed during preparation")
        state.update(
            canonical_input=str(canonical), canonical_sha256=s.sha256(canonical), stage="scoring"
        )
        checkpoint()
        batch_metadata = {
            "source_system": "ml_template.scoring.pipeline",
            "business_batch_id": identity,
        }
        s.score(
            engine,
            existing,
            canonical,
            check_only=True,
            chunk_size=chunk_size,
            batch_definition=batch_definition,
            batch_metadata=batch_metadata,
        )
        scored = s.score(
            engine,
            existing,
            canonical,
            check_only=False,
            chunk_size=chunk_size,
            batch_definition=batch_definition,
            batch_metadata=batch_metadata,
        )
        if scored.get("status") != "succeeded":
            raise ValueError("scoring did not succeed")
        predictions = Path(scored["artifact_dir"]) / "predictions.parquet"
        scoring_summary = s.read_json(Path(scored["artifact_dir"]) / "summary.json")
        if scoring_summary["input_sha256"] != state["canonical_sha256"]:
            raise ValueError("scorer snapshot differs from canonical pipeline input")
        state.update(
            status="succeeded",
            stage="complete",
            predictions_computed=True,
            scoring_batch_id=scored["scoring_batch_id"],
            scoring_dir=scored["artifact_dir"],
            predictions_path=str(predictions),
            predictions_sha256=s.sha256(predictions),
            n_rows=scored["n_rows"],
        )
        checkpoint()
        return state
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        raise


def _check_input(s, engine, release, source, input_format, chunk_size, batch_definition):
    if input_format == "raw":
        with contextlib.redirect_stdout(io.StringIO()):
            s.prepare_input(["--release", str(release), "--input", str(source), "--check"])
    else:
        s.score(
            engine,
            release,
            source,
            check_only=True,
            chunk_size=chunk_size,
            batch_definition=batch_definition,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-format", choices=["raw", "canonical"], default="raw")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--evaluation-id")
    evaluation.add_argument("--evaluation-config", type=Path)
    parser.add_argument(
        "--primary-metric", choices=["roc_auc", "pr_auc", "brier", "log_loss", "rmse", "mae"]
    )
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--acknowledge-prior-exposure", action="store_true")
    parser.add_argument("--prior-exposure-note", default="")
    parser.add_argument("--acknowledge-research-release", action="store_true")
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-definition")
    parser.add_argument(
        "--prep-dir", type=Path, help="Exact prepared-input directory; must not already exist"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    setup_logging()
    engine, services = (get_engine(), _services())
    spec = _evaluation_spec(
        services,
        engine,
        args.run_id,
        evaluation_id=args.evaluation_id,
        evaluation_config=args.evaluation_config,
        primary_metric=args.primary_metric,
        acknowledge_prior_exposure=args.acknowledge_prior_exposure,
        prior_exposure_note=args.prior_exposure_note,
        no_uncertainty=args.no_uncertainty,
    )
    result = run_pipeline(
        engine,
        run_id=args.run_id,
        input_path=args.input,
        evaluation_spec=spec,
        expected_evaluation_id=args.evaluation_id,
        release_dir=args.release_dir,
        input_format=args.input_format,
        output_root=args.output_root,
        chunk_size=args.chunk_size,
        batch_definition=args.batch_definition,
        check_only=args.check,
        prep_dir=args.prep_dir,
        acknowledge_research_release=args.acknowledge_research_release,
        services=services,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.check and (not result["ready"]):
        raise SystemExit(2)


def _services():
    s = _binary_services()
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.scoring.pipeline_regression import get_run_task, verify_existing_evaluation
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig

    s.get_run_task = get_run_task
    s.verify_existing_evaluation = verify_existing_evaluation
    s.RegressionEvaluationConfig = RegressionEvaluationConfig
    s.RegressionReleaseConfig = RegressionReleaseConfig
    return s


def _evaluation_spec(
    services,
    engine,
    run_id,
    *,
    evaluation_id=None,
    evaluation_config=None,
    primary_metric=None,
    acknowledge_prior_exposure=False,
    prior_exposure_note="",
    no_uncertainty=False,
):
    task = getattr(services, "get_run_task", lambda engine, run: "binary_classification")(
        engine, run_id
    )
    arguments = dict(
        evaluation_id=evaluation_id,
        evaluation_config=evaluation_config,
        primary_metric=primary_metric,
        acknowledge_prior_exposure=acknowledge_prior_exposure,
        prior_exposure_note=prior_exposure_note,
        no_uncertainty=no_uncertainty,
    )
    if task == "regression":
        from ml_template.scoring.pipeline_regression import regression_evaluation_spec

        return regression_evaluation_spec(services, engine, run_id, **arguments)
    if task != "binary_classification":
        raise ValueError(f"Unsupported pipeline task: {task!r}")
    return _binary_evaluation_spec(services, engine, run_id, **arguments)


def run_pipeline(
    engine,
    *,
    run_id,
    input_path,
    evaluation_spec,
    expected_evaluation_id=None,
    release_dir=None,
    input_format="raw",
    output_root=None,
    chunk_size=None,
    batch_definition=None,
    check_only=True,
    prep_dir=None,
    acknowledge_research_release=False,
    services=None,
):
    s = services or _services()
    task = getattr(s, "get_run_task", lambda engine, run: "binary_classification")(engine, run_id)
    arguments = dict(
        run_id=run_id,
        input_path=input_path,
        evaluation_spec=evaluation_spec,
        expected_evaluation_id=expected_evaluation_id,
        release_dir=release_dir,
        input_format=input_format,
        output_root=output_root,
        chunk_size=chunk_size,
        batch_definition=batch_definition,
        check_only=check_only,
        prep_dir=prep_dir,
        acknowledge_research_release=acknowledge_research_release,
        services=s,
    )
    if task == "regression":
        from ml_template.scoring.pipeline_regression import run_regression_pipeline

        return run_regression_pipeline(engine, **arguments)
    if task != "binary_classification":
        raise ValueError(f"Unsupported pipeline task: {task!r}")
    return _binary_run_pipeline(engine, **arguments)


if __name__ == "__main__":
    main()
