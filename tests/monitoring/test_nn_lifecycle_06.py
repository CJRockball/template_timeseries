"""Pure NN lineage, unit, maturity-support and report integrity tests."""

import copy
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_template.monitoring.nn_monitoring_contract import (
    assessment_scope,
    context_for,
    release_binding,
    validate_frozen_view,
    validate_saved_context,
    verify_pipeline,
)
from ml_template.usage_dashboard.nn_monitor_report import bundle_bytes, metric_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world(tmp_path, mode="canonical"):
    root = tmp_path / "nnpipe_unit"
    score = root / "scores/score1"
    score.mkdir(parents=True)
    (score / "predictions.parquet").write_bytes(b"metadata-only unit fixture")
    (score / "artifacts.json").write_text("{}")
    release = {
        "release_id": "rel_" + "0" * 24,
        "content_sha256": "c" * 64,
        "target_unit": "MPa",
        "parent_run_id": "run1",
        "evaluation_id": "eval1",
        "calibration_id": None,
        "release_variant": "raw",
        "predictor_kind": "final",
        "interval_columns": {},
    }
    binding = {
        "schema_version": 1,
        "release": release,
        "score": {
            "scoring_batch_id": "score1",
            "input_sha256": "a" * 64,
            "n_rows": 4,
            "predictions_sha256": sha(score / "predictions.parquet"),
            "artifacts_sha256": sha(score / "artifacts.json"),
        },
    }
    input_sha = binding["score"]["input_sha256"]
    if mode == "raw":
        prep = root / "prepared"
        prep.mkdir()
        (prep / "raw_input.csv").write_text("id,a\n1,2\n")
        (prep / "data.parquet").write_bytes(b"canonical fixture")
        input_sha = sha(prep / "raw_input.csv")
        binding["score"]["input_sha256"] = sha(prep / "data.parquet")
        (prep / "manifest.json").write_text(
            json.dumps(
                {
                    "input_sha256": input_sha,
                    "release_id": release["release_id"],
                    "canonical_sha256": binding["score"]["input_sha256"],
                }
            )
        )
    state = {
        "pipeline_id": root.name,
        "status": "succeeded",
        "stage": "complete",
        "predictions_computed": True,
        "labels_loaded": False,
        "evaluation_executed": False,
        "release_built": False,
        "task": "regression",
        "model_type": "nn",
        "release_id": release["release_id"],
        "release_content_sha256": release["content_sha256"],
        "target_unit": "MPa",
        "parent_run_id": "run1",
        "evaluation_id": "eval1",
        "calibration_id": None,
        "release_variant": "raw",
        "interval_columns": {},
        "scoring_batch_id": "score1",
        "predictions_sha256": binding["score"]["predictions_sha256"],
        "scoring_manifest_sha256": binding["score"]["artifacts_sha256"],
        "n_rows": 4,
        "scoring_dir": str(score),
        "request": {"input_format": mode},
        "input_sha256": input_sha,
    }
    (root / "pipeline.json").write_text(json.dumps(state))
    manifest(root)
    batch = SimpleNamespace(
        summary={"model_type": "nn", "target_unit": "MPa", "nn_binding": binding},
        record={"artifact_dir": str(score)},
    )
    config = SimpleNamespace(
        pipeline_dir=root,
        expected_pipeline_id=root.name,
        expected_pipeline_manifest_sha256=sha(root / "pipeline_manifest.json"),
        target_unit="MPa",
    )
    return (root, state, batch, config)


def manifest(root):
    entries = {
        p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in root.rglob("*")
        if p.is_file() and p.name != "pipeline_manifest.json"
    }
    (root / "pipeline_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": entries})
    )


@pytest.mark.parametrize("mode", ["raw", "canonical"])
def test_verified_lineage_without_prediction_decode(tmp_path, mode):
    root, _, batch, config = world(tmp_path, mode)
    ctx = context_for(batch, config)
    assert (
        ctx["pipeline_manifest_sha256"] == sha(root / "pipeline_manifest.json")
        and ctx["model_loaded"] is False
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("stage", "export"),
        ("target_unit", "wrong"),
        ("release_id", "wrong"),
        ("evaluation_id", "wrong"),
        ("calibration_id", "other"),
        ("scoring_batch_id", "other"),
        ("predictions_computed", False),
        ("labels_loaded", True),
        ("evaluation_executed", True),
        ("release_built", True),
        ("n_rows", 99),
    ],
)
def test_rehashed_wrong_envelope_identity_rejected(tmp_path, field, value):
    root, state, batch, config = world(tmp_path)
    state[field] = value
    (root / "pipeline.json").write_text(json.dumps(state))
    manifest(root)
    with pytest.raises(ValueError):
        verify_pipeline(root, root.name, sha(root / "pipeline_manifest.json"), batch)


@pytest.mark.parametrize(
    "kind",
    ["manifest", "prediction", "unlisted", "identity", "directory", "units", "missing_context"],
)
def test_context_failures(tmp_path, kind):
    root, _, batch, config = world(tmp_path)
    if kind == "manifest":
        config.expected_pipeline_manifest_sha256 = "f" * 64
    elif kind == "prediction":
        (root / "scores/score1/predictions.parquet").write_bytes(b"changed")
    elif kind == "unlisted":
        (root / "extra").write_text("x")
    elif kind == "identity":
        config.expected_pipeline_id = "nnpipe_other"
    elif kind == "directory":
        batch.record["artifact_dir"] = str(tmp_path / "other")
    elif kind == "units":
        config.target_unit = "wrong"
    else:
        config = SimpleNamespace(target_unit="MPa")
    with pytest.raises(ValueError):
        context_for(batch, config)


def test_reference_can_bind_another_batch_of_same_release(tmp_path):
    _, _, batch, config = world(tmp_path)
    ctx = context_for(batch, config)
    other = copy.deepcopy(batch)
    other.summary["nn_binding"]["score"]["scoring_batch_id"] = "score2"
    validate_saved_context({"nn_context": ctx}, other)
    with pytest.raises(ValueError):
        validate_saved_context({"nn_context": ctx}, other, same_batch=True)


def test_reference_cannot_cross_release(tmp_path):
    _, _, batch, config = world(tmp_path)
    ctx = context_for(batch, config)
    other = copy.deepcopy(batch)
    other.summary["nn_binding"]["release"]["target_unit"] = "other"
    with pytest.raises(ValueError):
        validate_saved_context({"nn_context": ctx}, other)


def test_missing_saved_context_rejected(tmp_path):
    _, _, batch, _ = world(tmp_path)
    with pytest.raises(ValueError):
        validate_saved_context({}, batch)


def test_legacy_classification_context_unchanged():
    assert (
        context_for(SimpleNamespace(summary={"task": "binary_classification"}), SimpleNamespace())
        is None
    )


def config():
    return SimpleNamespace(min_rows=200, min_labelled=100, min_label_coverage=0.9)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("none", "unknown_no_labels"),
        ("zero", "not_assessed_no_mature_observed_outcomes"),
        ("support", "not_assessed_insufficient_outcomes"),
        ("coverage", "not_assessed_label_coverage"),
        ("selected", "not_assessed_selected_or_unknown_sampling"),
        ("good", "assessed_mature_labelled_subset"),
    ],
)
def test_explicit_assessment_status(case, expected):
    perf = (
        None
        if case == "none"
        else {
            "n_evaluated": 0 if case == "zero" else 120,
            "sufficient_support": case != "support",
            "mature_label_coverage": 0.5 if case == "coverage" else 1.0,
        }
    )
    s = assessment_scope(config(), perf, "selected" if case == "selected" else "all_rows", 10, 10)
    assert (
        s["performance_status"] == expected
        and s["distribution_status"] == "not_assessed_insufficient_rows"
        and (s["automatic_promotion"] is False)
    )


def test_distribution_status_independent_of_label_status():
    s = assessment_scope(config(), None, None, 500, 300)
    assert (
        s["distribution_status"] == "descriptive" and s["performance_status"] == "unknown_no_labels"
    )


def frozen_view(tmp_path):
    _, _, batch, cfg = world(tmp_path)
    ctx = context_for(batch, cfg)
    c = {
        **vars(config()),
        "expected_pipeline_id": cfg.expected_pipeline_id,
        "expected_pipeline_manifest_sha256": cfg.expected_pipeline_manifest_sha256,
        "target_unit": "MPa",
    }
    summary = {
        "monitoring_id": "mon1",
        "model_type": "nn",
        "target_unit": "MPa",
        "release_id": ctx["nn_binding"]["release"]["release_id"],
        "scoring_batch_id": "score1",
        "nn_context": ctx,
        "n_rows": 4,
        "label_sampling": None,
        "assessment_scope": assessment_scope(config(), None, None, 4, 4),
    }
    return {
        "summary": summary,
        "config": c,
        "performance": None,
        "quality": {"n_rows": 4},
        "drift": [],
        "findings": [{"message": "<script>unsafe</script>"}],
    }


def test_frozen_scope_and_report_export(tmp_path):
    view = frozen_view(tmp_path)
    validate_frozen_view(view)
    data = bundle_bytes(view, "mon1")
    with zipfile.ZipFile(BytesIO(data)) as z:
        assert z.testzip() is None and "<script>unsafe" not in z.read("report.html").decode()
        checks = json.loads(z.read("SHA256SUMS.json"))
        assert all((hashlib.sha256(z.read(n)).hexdigest() == v for n, v in checks.items()))


@pytest.mark.parametrize("kind", ["unit", "pipeline", "scope", "batch"])
def test_frozen_scope_tampering(tmp_path, kind):
    view = frozen_view(tmp_path)
    if kind == "unit":
        view["summary"]["target_unit"] = "wrong"
    elif kind == "pipeline":
        view["config"]["expected_pipeline_id"] = "wrong"
    elif kind == "scope":
        view["summary"]["assessment_scope"]["performance_status"] = "healthy"
    else:
        view["summary"]["scoring_batch_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_frozen_view(view)


def test_regression_metrics_rows_do_not_require_class_counts():
    perf = {
        "task": "regression",
        "n_evaluated": 4,
        "sufficient_support": False,
        "score_metrics": {
            "y_pred_release": {
                "rmse": 1.0,
                "bias": -0.5,
                "r2": None,
                "residual_quantiles": {".5": 0.2},
            }
        },
        "interval_metrics": {
            "c800000": {
                "n_evaluated": 4,
                "nominal_coverage": 0.8,
                "empirical_coverage": 0.75,
                "mean_width": 3.0,
            }
        },
    }
    rows = metric_rows(perf)
    assert any(r["metric"] == "bias" and r["value"] == -0.5 for r in rows)
    assert any(r.get("level") == "c800000" for r in rows)
    assert all(r["sufficient_support"] is False for r in rows)


@pytest.mark.parametrize("kind", ["reference", "labels", "monitor", "template"])
def test_context_request_models(kind, tmp_path):
    from ml_template.monitoring.nn_monitor import (
        NNLabelConfig,
        NNMonitorConfig,
        NNReferenceConfig,
        NNTemplateConfig,
    )

    common = {
        "pipeline_dir": tmp_path,
        "expected_pipeline_id": "nnpipe_unit",
        "expected_pipeline_manifest_sha256": "a" * 64,
        "scoring_batch_id": "score1",
        "target_unit": "MPa",
    }
    if kind == "reference":
        obj = NNReferenceConfig(**common, name="unit", approved_population="Synthetic only")
    elif kind == "labels":
        obj = NNLabelConfig(
            **common,
            input_path="labels.csv",
            source="synthetic",
            outcome_definition="synthetic",
            sampling="all_rows",
        )
    elif kind == "monitor":
        obj = NNMonitorConfig(**common, reference_id="ref1", as_of="2026-01-03T00:00:00+00:00")
    else:
        obj = NNTemplateConfig(**common)
    assert obj.model_dump(mode="json")["expected_pipeline_manifest_sha256"] == "a" * 64


def test_context_requests_reject_missing_units():
    from ml_template.monitoring.nn_monitor import NNTemplateConfig

    with pytest.raises(ValueError):
        NNTemplateConfig(
            pipeline_dir="x",
            expected_pipeline_id="nnpipe_unit",
            expected_pipeline_manifest_sha256="a" * 64,
            scoring_batch_id="score1",
        )


def test_report_wrong_identity_rejected(tmp_path):
    with pytest.raises(ValueError):
        bundle_bytes(frozen_view(tmp_path), "other_monitor")


@pytest.mark.parametrize("kind", ["valid", "wrong_unit", "wrong_parent", "wrong_descriptor"])
def test_raw_nn_units_bound_from_release_without_mutating_score(tmp_path, kind):
    release = tmp_path / "release"
    release.mkdir()
    info = {
        "model_type": "nn",
        "task": "regression",
        "prediction_kind": "continuous",
        "target_unit": "MPa",
        "parent_run_id": "run1",
        "evaluation_id": "eval1",
        "calibration_id": None,
        "release_variant": "raw",
        "predictor_kind": "final",
        "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
    }
    summary = {
        "parent_run_id": "run1",
        "calibration_id": None,
        "release_variant": "raw",
        "prediction_columns": info["prediction_columns"],
    }
    descriptor = {"schema_version": 3, "model_type": "nn", "kind": "final"}
    if kind == "wrong_unit":
        summary["target_unit"] = "wrong"
    elif kind == "wrong_parent":
        summary["parent_run_id"] = "other"
    elif kind == "wrong_descriptor":
        descriptor["kind"] = "cv_mean"
    for name, value in [
        ("release.json", info),
        ("predictor.json", descriptor),
        ("runtime.json", {}),
    ]:
        (release / name).write_text(json.dumps(value))
    record = {
        "release_id": "rel1",
        "scoring_batch_id": "score1",
        "input_sha256": "a" * 64,
        "n_rows": 4,
    }
    before = copy.deepcopy(summary)
    if kind == "valid":
        result = release_binding(
            summary,
            info,
            {"content_sha256": "c" * 64},
            record,
            release,
            {"predictions.parquet": {"sha256": "b" * 64}},
            "d" * 64,
        )
        assert (
            result["target_unit"] == "MPa"
            and "target_unit" not in summary
            and (result["nn_binding"]["release"]["evaluation_id"] == "eval1")
        )
        assert result["model_type"] == "nn" and summary == before
    else:
        with pytest.raises(ValueError):
            release_binding(
                summary,
                info,
                {"content_sha256": "c" * 64},
                record,
                release,
                {"predictions.parquet": {"sha256": "b" * 64}},
                "d" * 64,
            )
