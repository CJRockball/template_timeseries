"""Regression orchestration only: delegates evaluation, packaging and scoring."""

import contextlib
import io
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def get_run_task(engine, run_id):
    from ml_template.db import model_run_registry
    from ml_template.tracking import run_artifacts as artifacts
    from ml_template.training.run_config import RunConfig, config_hash

    parent = model_run_registry.get_run(engine, run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("run_id must identify a successful saved model")
    directory = artifacts.resolve_stored_path(parent["artifact_dir"])
    artifacts.verify_artifact(directory, "config.json")
    cfg = RunConfig.model_validate_json((directory / "config.json").read_text())
    if config_hash(cfg) != parent["config_hash"]:
        raise ValueError("Saved run config/registry identity differs")
    return cfg.task


def _normalised_spec(spec, saved):
    payload = spec.model_dump(mode="json")
    actual = saved.model_dump(mode="json")
    for name in [
        "expected_silver_dataset_version_id",
        "expected_feature_set_id",
        "expected_split_id",
    ]:
        if payload.get(name) is None:
            payload[name] = actual.get(name)
    return type(saved).model_validate(payload)


def verify_existing_evaluation(engine, run_id, evaluation_id, spec):
    """Reuse explicit evidence by artifact/prediction identity, not current code-cache key.

    regression_evidence checks saved spec vs registry, the evidence digest, original
    model/data hashes and evaluated prediction code. No target values are loaded.
    """
    from ml_template.scoring.regression_release import regression_evidence

    _prepared, directory, saved, summary, provenance = regression_evidence(
        engine,
        SimpleNamespace(
            parent_run_id=run_id,
            evaluation_id=evaluation_id,
            calibration_id=getattr(spec, "calibration_id", None),
        ),
    )
    if _normalised_spec(spec, saved) != saved:
        raise ValueError("Requested specification differs from the frozen evaluation")
    return {
        "status": "succeeded",
        "cached": False,
        "reused": True,
        "reuse_mode": "verified_explicit_evaluation",
        "evaluation_id": evaluation_id,
        "artifact_dir": str(directory),
        "identity_hash": summary["identity_hash"],
        "summary": summary,
        "holdout_labels_loaded": False,
        "_predictor": provenance["predictor"],
        "_config_sha256": provenance["input_hashes"]["config"],
        "_evaluation_spec": saved.model_dump(mode="json"),
        "_evaluation_provenance": provenance,
    }


def regression_evaluation_spec(
    s,
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
        raise ValueError("Supply evaluation_id OR evaluation_config, not both")
    overrides = bool(
        primary_metric or acknowledge_prior_exposure or prior_exposure_note or no_uncertainty
    )
    if evaluation_id:
        if overrides:
            raise ValueError(
                "An existing evaluation ID freezes its spec; omit evaluation overrides"
            )
        record = s.get_evaluation(engine, evaluation_id)
        if not record or record["status"] != "succeeded" or record["parent_run_id"] != run_id:
            raise ValueError("Evaluation ID must identify a successful evaluation of this run")
        spec = s.RegressionEvaluationConfig.model_validate_json(record["spec_json"])
    elif evaluation_config:
        if overrides:
            raise ValueError("Put evaluation settings in the supplied evaluation YAML")
        spec = s.load_evaluation_config(evaluation_config)
    else:
        payload = {
            "parent_run_id": run_id,
            "primary_metric": primary_metric or "rmse",
            "uncertainty": {"enabled": False},
        }
        if acknowledge_prior_exposure or prior_exposure_note:
            if not acknowledge_prior_exposure or not prior_exposure_note.strip():
                raise ValueError("Prior exposure requires both explicit acknowledgement and a note")
            payload.update(acknowledge_prior_exposure=True, prior_exposure_note=prior_exposure_note)
        spec = s.RegressionEvaluationConfig.model_validate(payload)
    if getattr(spec, "task", None) != "regression" or spec.parent_run_id != run_id:
        raise ValueError("Evaluation task/parent differs from saved regression run")
    return spec


def verify_regression_release(s, directory, run_id, spec, binding):
    directory = Path(directory).resolve()
    manifest = s.verify(directory)
    try:
        s.verify_runtime(s.read_json(directory / "runtime.json"))
    except ValueError as exc:
        raise ValueError(
            "Release runtime is pinned to different code/environment. Build a NEW release from the SAME verified evaluation; do not edit/reseal the old release or reevaluate the holdout merely for orchestration changes."
        ) from exc
    info = s.read_json(directory / "release.json")
    if (
        info.get("schema_version") != 2
        or info.get("task") != "regression"
        or info.get("prediction_kind") != "continuous"
        or info["parent_run_id"] != run_id
        or info["evaluation_id"] != binding["evaluation_id"]
        or info.get("calibration_id") != spec.calibration_id
        or info.get("release_variant") != spec.release_variant
        or info.get("policy", {}).get("kind") != "none"
        or info.get("predictor_kind") != spec.predictor.kind
    ):
        raise ValueError("Existing release differs from the frozen regression predictor/evaluation")
    descriptor = s.read_json(directory / "predictor.json")
    evidence = s.read_json(directory / "evidence.json")
    if (
        descriptor != binding["_predictor"]
        or evidence["provenance"] != binding["_evaluation_provenance"]
    ):
        raise ValueError("Release predictor/evidence differs from verified evaluation")
    if (
        evidence["evaluation_spec"] != binding["_evaluation_spec"]
        or evidence["evaluation_summary"] != binding["summary"]
    ):
        raise ValueError("Release evaluation specification/summary differs")
    if manifest["files"].get("config.json", {}).get("sha256") != binding["_config_sha256"]:
        raise ValueError("Packaged run configuration differs from evaluated configuration")
    if info.get("prediction_columns") != {"raw": "y_pred_raw", "release": "y_pred_release"}:
        raise ValueError("Unexpected regression output schema")
    return manifest


def run_regression_pipeline(
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
    services,
):
    from ml_template.scoring.pipeline import _check_input

    s = services
    source = Path(input_path).resolve()
    if not source.is_file() or source.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Input must be an existing CSV or Parquet file")
    if input_format not in {"raw", "canonical"}:
        raise ValueError("input_format must be raw or canonical")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    parent = s.get_run(engine, run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("run_id must identify a successful saved model")
    if parent.get("model_type") == "nn" and (release_dir is None or expected_evaluation_id is None):
        raise ValueError(
            "NN operational scoring requires an explicit existing release and evaluation; use the release-first YAML pipeline"
        )
    if (
        getattr(evaluation_spec, "task", None) != "regression"
        or evaluation_spec.parent_run_id != run_id
    ):
        raise ValueError("Evaluation task/parent differs from saved regression run")
    if batch_definition is not None:
        raise ValueError("Regression has no classification policy; omit --batch-definition")
    if prep_dir is not None:
        if input_format != "raw":
            raise ValueError("--prep-dir is only used with raw input")
        prep_dir = Path(prep_dir).expanduser().resolve()
        if prep_dir.exists():
            raise FileExistsError(f"Prepared-input directory already exists: {prep_dir}")
    if release_dir is not None and expected_evaluation_id is None:
        raise ValueError("Regression release reuse requires explicit --evaluation-id")
    if not check_only and not acknowledge_research_release:
        raise ValueError("Execution requires --acknowledge-research-release")
    before = s.sha256(source)
    if expected_evaluation_id:
        preflight = s.verify_existing_evaluation(
            engine, run_id, expected_evaluation_id, evaluation_spec
        )
        if (
            preflight.get("status") != "succeeded"
            or preflight.get("evaluation_id") != expected_evaluation_id
        ):
            raise ValueError("Explicit evaluation verification returned the wrong identity/status")
        reuse = True
    else:
        preflight = s.evaluate(engine, evaluation_spec, check_only=True)
        reuse = preflight.get("status") == "succeeded" and bool(preflight.get("cached"))
    ready = reuse or bool(preflight.get("ready"))
    visible = {key: value for key, value in preflight.items() if not key.startswith("_")}
    report = {
        "schema_version": 2,
        "task": "regression",
        "prediction_kind": "continuous",
        "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
        "parent_run_id": run_id,
        "input_path": str(source),
        "input_sha256": before,
        "input_format": input_format,
        "evaluation_ready": ready,
        "evaluation_preflight": visible,
        "evaluation_reused": reuse,
        "downstream_checks": "deferred_until_release_exists",
        "predictions_computed": False,
    }
    existing = None
    if release_dir is not None:
        existing = Path(release_dir).resolve()
        manifest = verify_regression_release(s, existing, run_id, evaluation_spec, preflight)
        report.update(release_id=manifest["release_id"], release_dir=str(existing))
        _check_input(s, engine, existing, source, input_format, chunk_size, None)
        report["downstream_checks"] = "input_checked; prediction_execution_not_run"
    if s.sha256(source) != before:
        raise ValueError("Input changed during preflight")
    if check_only:
        return {**report, "status": "checked", "ready": ready, "pipeline_artifacts_written": False}
    if not ready:
        raise ValueError(
            "Evaluation preflight blocked; review prior exposure and resolved spec. For an existing successful evaluation, use its explicit --evaluation-id; this pipeline never silently reevaluates it."
        )
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "pipelines"
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
        evaluated = preflight if reuse else s.evaluate(engine, evaluation_spec, check_only=False)
        if evaluated.get("status") != "succeeded":
            raise ValueError("Evaluation did not succeed")
        eid = evaluated["evaluation_id"]
        binding = (
            preflight
            if expected_evaluation_id
            else s.verify_existing_evaluation(engine, run_id, eid, evaluation_spec)
        )
        state.update(
            evaluation_id=eid,
            evaluation_dir=evaluated["artifact_dir"],
            evaluation_reuse_mode=(
                binding["reuse_mode"]
                if expected_evaluation_id
                else "native_cache"
                if reuse
                else "new_evaluation_executed"
            ),
            stage="release",
        )
        checkpoint()
        if existing is None:
            request = s.RegressionReleaseConfig(
                name=f"pipeline_{identity}",
                evaluation_id=eid,
                parent_run_id=run_id,
                calibration_id=evaluation_spec.calibration_id,
                intended_use=evaluation_spec.intended_use,
                acknowledge_research_release=True,
                parity_atol=1e-7,
            )
            checked = s.build_release(engine, request, check_only=True)
            if not checked.get("ready"):
                raise ValueError("Release preflight is not ready")
            built = s.build_release(engine, request, check_only=False)
            if built.get("status") != "succeeded":
                raise ValueError("Release creation did not succeed")
            existing = Path(built["artifact_dir"]).resolve()
            manifest = verify_regression_release(s, existing, run_id, evaluation_spec, binding)
            if manifest["release_id"] != built["release_id"]:
                raise ValueError("Returned release identity differs from package")
            state.update(release_id=built["release_id"], release_dir=str(existing))
        state["stage"] = "input_preparation"
        checkpoint()
        if s.sha256(source) != before:
            raise ValueError("Input changed after evaluation/release")
        _check_input(s, engine, existing, source, input_format, chunk_size, None)
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
            if (
                preparation["input_sha256"] != before
                or preparation.get("release_id") != state["release_id"]
            ):
                raise ValueError("Prepared snapshot/release differs from requested input")
            if preparation.get("canonical_sha256") != s.sha256(canonical):
                raise ValueError("Prepared canonical artifact hash differs from manifest")
        if s.sha256(source) != before:
            raise ValueError("Input changed during preparation")
        state.update(
            canonical_input=str(canonical), canonical_sha256=s.sha256(canonical), stage="scoring"
        )
        checkpoint()
        metadata = {"source_system": "ml_template.scoring.pipeline", "business_batch_id": identity}
        s.score(
            engine,
            existing,
            canonical,
            check_only=True,
            chunk_size=chunk_size,
            batch_definition=None,
            batch_metadata=metadata,
        )
        scored = s.score(
            engine,
            existing,
            canonical,
            check_only=False,
            chunk_size=chunk_size,
            batch_definition=None,
            batch_metadata=metadata,
        )
        if scored.get("status") != "succeeded":
            raise ValueError("Scoring did not succeed")
        score_dir = Path(scored["artifact_dir"])
        summary = s.read_json(score_dir / "summary.json")
        if (
            summary["input_sha256"] != state["canonical_sha256"]
            or summary["release_id"] != state["release_id"]
        ):
            raise ValueError("Scorer snapshot/release differs from pipeline input")
        if summary.get("task") != "regression" or summary.get("prediction_kind") != "continuous":
            raise ValueError("Scorer output has wrong task semantics")
        if s.sha256(source) != before:
            raise ValueError("Input changed during scoring")
        predictions = score_dir / "predictions.parquet"
        state.update(
            status="succeeded",
            stage="complete",
            predictions_computed=True,
            scoring_batch_id=scored["scoring_batch_id"],
            scoring_dir=str(score_dir),
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
