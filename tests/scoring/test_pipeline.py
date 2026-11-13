"""Orchestration tests with mocked services; no real DB, holdout or model."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_template.scoring.pipeline import run_pipeline


class Policy:
    kind = "none"
    batch_definition = "entire evaluation holdout"

    def model_dump(self, **kwargs):
        return {"kind": self.kind}


class Spec:
    parent_run_id = "run_one"
    calibration_id = None
    release_variant = "raw"
    intended_use = "Research only"
    policy = Policy()

    def model_dump(self, **kwargs):
        return {"parent_run_id": self.parent_run_id}


@pytest.fixture
def environment(tmp_path):
    source = tmp_path / "test.csv"
    source.write_text("id,x\n1,2\n")
    calls = []
    options = {"cached": False, "ready": True, "score_fails": False}

    def write_json(path, value):
        Path(path).write_text(json.dumps(value))

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def evaluate(engine, spec, *, check_only):
        calls.append(("evaluate", check_only))
        if check_only and not options["cached"]:
            return {"ready": options["ready"], "holdout_labels_loaded": False}
        return {
            "evaluation_id": "eval_one",
            "artifact_dir": str(tmp_path / "evaluation"),
            "status": "succeeded",
            "cached": options["cached"],
        }

    def build_release(engine, request, *, check_only):
        calls.append(("release", check_only))
        folder = tmp_path / "release"
        folder.mkdir(exist_ok=True)
        return (
            {"ready": True}
            if check_only
            else {"status": "succeeded", "release_id": "rel_one", "artifact_dir": str(folder)}
        )

    def prepare_input(argv):
        calls.append(("prepare", "--check" in argv))
        if "--execute" in argv:
            directory = Path(argv[argv.index("--output-dir") + 1])
            directory.mkdir()
            (directory / "data.parquet").write_bytes(b"canonical")
            write_json(directory / "manifest.json", {"input_sha256": sha256(source)})

    def score(engine, release, canonical, *, check_only, **kwargs):
        calls.append(("score", check_only))
        if options["score_fails"]:
            raise ValueError("synthetic scorer failure")
        if check_only:
            return {"ready": True}
        directory = tmp_path / "score"
        directory.mkdir(exist_ok=True)
        (directory / "predictions.parquet").write_bytes(b"predictions")
        write_json(directory / "summary.json", {"input_sha256": sha256(canonical)})
        return {
            "status": "succeeded",
            "artifact_dir": str(directory),
            "scoring_batch_id": "score_one",
            "n_rows": 1,
        }

    s = SimpleNamespace(
        get_run=lambda engine, run: {"status": "succeeded"},
        evaluate=evaluate,
        build_release=build_release,
        ReleaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        prepare_input=prepare_input,
        score=score,
        sha256=sha256,
        write_json=write_json,
        read_json=lambda path: json.loads(Path(path).read_text()),
    )
    return SimpleNamespace(
        source=source, s=s, calls=calls, options=options, output=tmp_path / "pipelines", spec=Spec()
    )


def run(env, **kwargs):
    options = {
        "run_id": "run_one",
        "input_path": env.source,
        "evaluation_spec": env.spec,
        "output_root": env.output,
        "services": env.s,
    }
    options.update(kwargs)
    return run_pipeline(object(), **options)


def test_preflight_never_executes_or_writes_pipeline(environment):
    result = run(environment)
    assert result["ready"] is True
    assert result["downstream_checks"] == "deferred_until_release_exists"
    assert environment.calls == [("evaluate", True)]
    assert not environment.output.exists()


def test_full_raw_pipeline_and_returned_path(environment):
    result = run(environment, check_only=False, acknowledge_research_release=True)
    assert environment.calls == [
        ("evaluate", True),
        ("evaluate", False),
        ("release", True),
        ("release", False),
        ("prepare", True),
        ("prepare", False),
        ("score", True),
        ("score", False),
    ]
    assert result["status"] == "succeeded"
    assert result["evaluation_id"] == "eval_one"
    assert result["release_id"] == "rel_one"
    assert result["scoring_batch_id"] == "score_one"
    assert Path(result["predictions_path"]).exists()
    saved = json.loads((Path(result["artifact_dir"]) / "pipeline.json").read_text())
    assert saved["predictions_sha256"] == environment.s.sha256(result["predictions_path"])


def test_cached_evaluation_not_reexecuted(environment):
    environment.options["cached"] = True
    run(environment, check_only=False, acknowledge_research_release=True)
    assert ("evaluate", False) not in environment.calls


def test_blocked_preflight_cannot_proceed(environment):
    environment.options["ready"] = False
    with pytest.raises(ValueError, match="blocked"):
        run(environment, check_only=False, acknowledge_research_release=True)
    assert environment.calls == [("evaluate", True)]
    assert not environment.output.exists()


def test_check_reports_not_ready(environment):
    environment.options["ready"] = False
    assert run(environment)["ready"] is False


def test_research_acknowledgement_required(environment):
    with pytest.raises(ValueError, match="acknowledge-research-release"):
        run(environment, check_only=False)
    assert not environment.calls


def test_wrong_evaluation_id_stops(environment):
    environment.options["cached"] = True
    with pytest.raises(ValueError, match="no longer matches"):
        run(environment, expected_evaluation_id="eval_other")


def test_failed_scoring_is_checkpointed(environment):
    environment.options["score_fails"] = True
    with pytest.raises(ValueError, match="synthetic"):
        run(environment, check_only=False, acknowledge_research_release=True)
    state = json.loads(next(environment.output.glob("*/pipeline.json")).read_text())
    assert state["status"] == "failed"
    assert state["stage"] == "scoring"
    assert "predictions_path" not in state


def test_canonical_input_skips_preparation(environment):
    result = run(
        environment, check_only=False, acknowledge_research_release=True, input_format="canonical"
    )
    assert result["canonical_input"] == str(environment.source.resolve())
    assert not any(stage == "prepare" for stage, _ in environment.calls)


def test_bad_run_rejected(environment):
    environment.s.get_run = lambda engine, run: {"status": "failed"}
    with pytest.raises(ValueError, match="successful saved model"):
        run(environment)


def test_missing_input_rejected(environment):
    with pytest.raises(ValueError, match="existing CSV"):
        run(environment, input_path=environment.source.parent / "absent.csv")


def test_parent_mismatch_rejected(environment):
    with pytest.raises(ValueError, match="parent"):
        run(environment, run_id="different")


def test_input_change_after_evaluation_stops(environment):
    original = environment.s.build_release

    def build(engine, request, *, check_only):
        result = original(engine, request, check_only=check_only)
        if not check_only:
            environment.source.write_text("id,x\n1,999\n")
        return result

    environment.s.build_release = build
    with pytest.raises(ValueError, match="input changed"):
        run(environment, check_only=False, acknowledge_research_release=True)
    assert not any(stage == "score" for stage, _ in environment.calls)


def existing_release(env, *, parent="run_one", eid="eval_one"):
    folder = env.source.parent / "existing_release"
    folder.mkdir()
    info = {
        "parent_run_id": parent,
        "evaluation_id": eid,
        "calibration_id": None,
        "release_variant": "raw",
        "policy": {"kind": "none"},
    }
    (folder / "release.json").write_text(json.dumps(info))
    (folder / "runtime.json").write_text("{}")
    env.s.verify = lambda path: {"release_id": "rel_existing"}
    env.s.verify_runtime = lambda runtime: None
    env.options["cached"] = True
    return folder


def test_explicit_existing_release_reused(environment):
    folder = existing_release(environment)
    result = run(
        environment, release_dir=folder, check_only=False, acknowledge_research_release=True
    )
    assert result["release_id"] == "rel_existing"
    assert not any(stage == "release" for stage, _ in environment.calls)


def test_wrong_release_parent_rejected(environment):
    folder = existing_release(environment, parent="another_run")
    with pytest.raises(ValueError, match="differs"):
        run(environment, release_dir=folder)


def test_wrong_release_evaluation_rejected(environment):
    folder = existing_release(environment, eid="another_eval")
    with pytest.raises(ValueError, match="cached evaluation"):
        run(environment, release_dir=folder)


def test_pipeline_metadata_matches_real_schema(environment):
    from ml_template.monitoring.scoring_attempts import parse_metadata

    original = environment.s.score
    captured = []

    def strict_score(*args, **kwargs):
        captured.append(parse_metadata(kwargs.get("batch_metadata")))
        return original(*args, **kwargs)

    environment.s.score = strict_score

    result = run(
        environment,
        check_only=False,
        acknowledge_research_release=True,
    )

    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert captured[0]["business_batch_id"] == result["pipeline_id"]
