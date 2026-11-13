"""Regression pipeline service tests: no model fitting or holdout labels."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_template.evaluation.regression_config import RegressionEvaluationConfig
from ml_template.scoring.pipeline import _evaluation_spec, run_pipeline
from ml_template.scoring.pipeline_regression import verify_existing_evaluation


@pytest.fixture
def env(tmp_path):
    source = tmp_path / "unlabelled.csv"
    source.write_text("id,x\n1,2\n")
    calls, metadata = [], []
    options = {
        "ready": True,
        "native_cached": False,
        "score_fails": False,
        "stale_runtime": False,
        "change_input": False,
        "evaluation_exists": True,
    }
    spec = RegressionEvaluationConfig(parent_run_id="run_one", intended_use="Research test")
    saved = spec.model_copy(
        update={
            "expected_silver_dataset_version_id": "silver_one",
            "expected_split_id": "split_one",
        }
    )
    descriptor = {
        "schema_version": 2,
        "kind": "final",
        "task": "regression",
        "prediction_kind": "continuous",
    }
    provenance = {"predictor": descriptor, "input_hashes": {"config": "a" * 64}, "code_hashes": {}}
    summary = {
        "evaluation_id": "eval_one",
        "parent_run_id": "run_one",
        "identity_hash": "old_evaluation_identity",
        "task": "regression",
    }
    binding = {
        "status": "succeeded",
        "cached": False,
        "reused": True,
        "reuse_mode": "verified_explicit_evaluation",
        "evaluation_id": "eval_one",
        "artifact_dir": str(tmp_path / "evaluation"),
        "summary": summary,
        "holdout_labels_loaded": False,
        "_predictor": descriptor,
        "_config_sha256": "a" * 64,
        "_evaluation_spec": saved.model_dump(mode="json"),
        "_evaluation_provenance": provenance,
    }

    def write_json(path, data):
        Path(path).write_text(json.dumps(data))

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def make_release():
        path = tmp_path / "release"
        path.mkdir(exist_ok=True)
        info = {
            "schema_version": 2,
            "task": "regression",
            "prediction_kind": "continuous",
            "parent_run_id": "run_one",
            "evaluation_id": "eval_one",
            "calibration_id": None,
            "release_variant": "raw",
            "policy": {"kind": "none"},
            "predictor_kind": "final",
            "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
        }
        write_json(path / "release.json", info)
        write_json(path / "runtime.json", {})
        write_json(path / "predictor.json", descriptor)
        write_json(
            path / "evidence.json",
            {
                "provenance": provenance,
                "evaluation_spec": saved.model_dump(mode="json"),
                "evaluation_summary": summary,
            },
        )
        return path

    def verify_runtime(value):
        if options["stale_runtime"]:
            raise ValueError("runtime/code differs from release")

    def verify_evaluation(engine, rid, eid, provided):
        calls.append(("verified_evaluation", eid))
        if rid != "run_one" or eid != "eval_one" or not options["evaluation_exists"]:
            raise ValueError("Invalid explicit evaluation identity")
        return copy.deepcopy(binding)

    def evaluate(engine, provided, *, check_only):
        calls.append(("evaluate", check_only))
        if check_only and not options["native_cached"]:
            return {"ready": options["ready"], "holdout_labels_loaded": False}
        return {
            "status": "succeeded",
            "cached": options["native_cached"],
            "evaluation_id": "eval_one",
            "artifact_dir": str(tmp_path / "evaluation"),
        }

    def build(engine, request, *, check_only):
        calls.append(("release", check_only))
        assert request.acknowledge_research_release and request.parent_run_id == "run_one"
        if check_only:
            return {"ready": True}
        path = make_release()
        if options["change_input"]:
            source.write_text("id,x\n1,999\n")
        return {"status": "succeeded", "release_id": "rel_one", "artifact_dir": str(path)}

    def prepare(argv):
        calls.append(("prepare", "--check" in argv))
        if "--execute" in argv:
            path = Path(argv[argv.index("--output-dir") + 1])
            path.mkdir()
            (path / "data.parquet").write_bytes(b"canonical")
            write_json(
                path / "manifest.json",
                {
                    "input_sha256": sha(source),
                    "release_id": "rel_one",
                    "canonical_sha256": sha(path / "data.parquet"),
                },
            )

    def score(engine, release, canonical, *, check_only, batch_metadata=None, **kwargs):
        calls.append(("score", check_only))
        metadata.append(copy.deepcopy(batch_metadata))
        if check_only:
            return {"ready": True}
        if options["score_fails"]:
            raise RuntimeError("synthetic regression scoring failure")
        path = tmp_path / "score"
        path.mkdir(exist_ok=True)
        (path / "predictions.parquet").write_bytes(b"regression predictions")
        write_json(
            path / "summary.json",
            {
                "input_sha256": sha(canonical),
                "release_id": "rel_one",
                "task": "regression",
                "prediction_kind": "continuous",
            },
        )
        return {
            "status": "succeeded",
            "artifact_dir": str(path),
            "scoring_batch_id": "score_one",
            "n_rows": 1,
        }

    services = SimpleNamespace(
        get_run_task=lambda e, r: "regression",
        get_run=lambda e, r: {"status": "succeeded"},
        get_evaluation=lambda e, eid: {
            "status": "succeeded",
            "parent_run_id": "run_one",
            "spec_json": saved.model_dump_json(),
        },
        RegressionEvaluationConfig=RegressionEvaluationConfig,
        RegressionReleaseConfig=lambda **kw: SimpleNamespace(**kw),
        verify_existing_evaluation=verify_evaluation,
        evaluate=evaluate,
        build_release=build,
        prepare_input=prepare,
        score=score,
        verify=lambda path: {
            "release_id": "rel_one",
            "files": {"config.json": {"sha256": "a" * 64}},
        },
        verify_runtime=verify_runtime,
        read_json=lambda path: json.loads(Path(path).read_text()),
        write_json=write_json,
        sha256=sha,
        load_evaluation_config=lambda path: saved,
    )
    return SimpleNamespace(
        source=source,
        output=tmp_path / "pipelines",
        calls=calls,
        options=options,
        services=services,
        spec=spec,
        saved=saved,
        binding=binding,
        metadata=metadata,
        make_release=make_release,
    )


def run(env, **kwargs):
    args = dict(
        run_id="run_one",
        input_path=env.source,
        evaluation_spec=env.spec,
        expected_evaluation_id="eval_one",
        input_format="canonical",
        output_root=env.output,
        services=env.services,
    )
    args.update(kwargs)
    return run_pipeline(object(), **args)


def test_explicit_evaluation_check_never_reevaluates_or_writes(env):
    result = run(env)
    assert result["ready"] and result["task"] == "regression"
    assert env.calls == [("verified_evaluation", "eval_one")]
    assert not env.output.exists()
    assert result["downstream_checks"] == "deferred_until_release_exists"


def test_explicit_evaluation_new_release_and_canonical_scoring(env):
    result = run(env, check_only=False, acknowledge_research_release=True)
    assert (
        result["status"] == "succeeded"
        and result["evaluation_reuse_mode"] == "verified_explicit_evaluation"
    )
    assert not any(name == "evaluate" for name, _ in env.calls)
    assert not any(name == "prepare" for name, _ in env.calls)
    state = json.loads((Path(result["artifact_dir"]) / "pipeline.json").read_text())
    assert state["prediction_columns"]["release"] == "y_pred_release"
    assert state["predictions_sha256"] == env.services.sha256(state["predictions_path"])
    assert env.metadata[-2] == env.metadata[-1]
    assert set(env.metadata[-1]) == {"source_system", "business_batch_id"}
    assert env.metadata[-1]["business_batch_id"] == result["pipeline_id"]


def test_raw_input_and_exact_prep_directory(env, tmp_path):
    prepared = tmp_path / "prepared"
    result = run(
        env,
        check_only=False,
        acknowledge_research_release=True,
        input_format="raw",
        prep_dir=prepared,
    )
    assert result["canonical_input"] == str(prepared / "data.parquet")
    assert ("prepare", False) in env.calls


def test_existing_release_reused_without_rebuild_or_evaluation(env):
    release = env.make_release()
    result = run(env, release_dir=release, check_only=False, acknowledge_research_release=True)
    assert result["release_id"] == "rel_one"
    assert not any(name in {"release", "evaluate"} for name, _ in env.calls)


def test_pinned_old_runtime_is_not_bypassed(env):
    release = env.make_release()
    env.options["stale_runtime"] = True
    with pytest.raises(ValueError, match="SAME verified evaluation"):
        run(env, release_dir=release)
    assert not env.output.exists()
    assert not any(name in {"release", "evaluate"} for name, _ in env.calls)


def test_release_predictor_mismatch_is_rejected(env):
    release = env.make_release()
    info = env.services.read_json(release / "release.json")
    info["predictor_kind"] = "cv_mean"
    env.services.write_json(release / "release.json", info)
    with pytest.raises(ValueError, match="differs"):
        run(env, release_dir=release)


def test_missing_native_cache_never_silently_reuses_or_acks_holdout(env):
    env.options["ready"] = False
    with pytest.raises(ValueError, match="blocked"):
        run(env, expected_evaluation_id=None, check_only=False, acknowledge_research_release=True)
    assert env.calls == [("evaluate", True)]
    assert not env.output.exists()


def test_new_evaluation_only_runs_after_ready_and_execute(env):
    result = run(
        env, expected_evaluation_id=None, check_only=False, acknowledge_research_release=True
    )
    assert result["evaluation_reuse_mode"] == "new_evaluation_executed"
    assert ("evaluate", True) in env.calls and ("evaluate", False) in env.calls


def test_failure_checkpoint_and_no_export(env):
    env.options["score_fails"] = True
    with pytest.raises(RuntimeError, match="synthetic"):
        run(env, check_only=False, acknowledge_research_release=True)
    state = json.loads(next(env.output.glob("*/pipeline.json")).read_text())
    assert state["status"] == "failed" and state["stage"] == "scoring"
    assert "predictions_path" not in state


def test_input_mutation_stops_before_scoring(env):
    env.options["change_input"] = True
    with pytest.raises(ValueError, match="Input changed"):
        run(env, check_only=False, acknowledge_research_release=True)
    assert not any(name == "score" for name, _ in env.calls)


@pytest.mark.parametrize(
    "extra",
    [
        {"chunk_size": 0},
        {"chunk_size": True},
        {"chunk_size": 1.5},
        {"batch_definition": "some policy"},
        {"prep_dir": "unused"},
        {"release_dir": "unused", "expected_evaluation_id": None},
    ],
)
def test_bad_arguments_rejected_before_evaluation(env, extra):
    with pytest.raises((ValueError, FileExistsError)):
        run(env, **extra)
    assert not env.calls


def test_frozen_spec_and_defaults(env):
    spec = _evaluation_spec(env.services, object(), "run_one", evaluation_id="eval_one")
    assert spec == env.saved
    with pytest.raises(ValueError, match="freezes"):
        _evaluation_spec(
            env.services, object(), "run_one", evaluation_id="eval_one", primary_metric="mae"
        )
    default = _evaluation_spec(env.services, object(), "run_one")
    assert default.primary_metric == "rmse" and not default.uncertainty.enabled
    with pytest.raises(ValueError):
        _evaluation_spec(env.services, object(), "run_one", primary_metric="roc_auc")


def test_verified_evidence_normalises_expected_ids_but_not_recipe(env, monkeypatch):
    from ml_template.scoring import regression_release

    monkeypatch.setattr(
        regression_release,
        "regression_evidence",
        lambda *args: (
            None,
            Path("evaluation"),
            env.saved,
            env.binding["summary"],
            env.binding["_evaluation_provenance"],
        ),
    )
    checked = verify_existing_evaluation(object(), "run_one", "eval_one", env.spec)
    assert checked["reused"] and not checked["cached"] and checked["holdout_labels_loaded"] is False
    different = env.spec.model_copy(update={"primary_metric": "mae"})
    with pytest.raises(ValueError, match="differs"):
        verify_existing_evaluation(object(), "run_one", "eval_one", different)


def test_binary_dispatch_retains_legacy_path(env, monkeypatch):
    from ml_template.scoring import pipeline

    env.services.get_run_task = lambda e, r: "binary_classification"
    marker = {"binary_legacy": True}
    monkeypatch.setattr(pipeline, "_binary_run_pipeline", lambda *a, **kw: marker)
    assert run(env) is marker
    assert not env.calls


def test_execution_requires_research_ack_before_evaluation(env):
    with pytest.raises(ValueError, match="acknowledge-research-release"):
        run(env, check_only=False)
    assert not env.calls


def test_native_cache_reuse_does_not_execute_evaluation(env):
    env.options["native_cached"] = True
    result = run(
        env, expected_evaluation_id=None, check_only=False, acknowledge_research_release=True
    )
    assert result["evaluation_reuse_mode"] == "native_cache"
    assert ("evaluate", True) in env.calls and ("evaluate", False) not in env.calls
