"""Native NN scored artifacts -> reference/labels/monitor -> read-only usage/report."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).parents[1] / "scoring" / "test_nn_lifecycle_05_native.py"
_spec = importlib.util.spec_from_file_location("nn_package_06_training_fixture", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
trained = _module.trained
_microbatch_before_training = _module._microbatch_before_synthetic_training


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
@pytest.mark.parametrize("corrected", [False, True], ids=["raw", "bias_intervals"])
def test_native_nn_monitoring(trained, tmp_path, monkeypatch, kind, corrected):
    from ml_template.monitoring.artifacts import digest
    from ml_template.monitoring.monitor_inputs import load_batch
    from ml_template.monitoring.nn_monitor import (
        NNLabelConfig,
        NNMonitorConfig,
        NNReferenceConfig,
        NNTemplateConfig,
        run,
    )
    from ml_template.monitoring.nn_monitoring_contract import context_for
    from ml_template.scoring.nn_batch_pipeline import BatchRequest, run_pipeline
    from ml_template.usage_dashboard.monitoring_data import load_assessment, metric_rows
    from ml_template.usage_dashboard.monitoring_page import monitoring_payload
    from ml_template.usage_dashboard.nn_monitor_report import bundle_bytes

    release, root = _module.build(trained, tmp_path, kind, corrected)
    frame = pd.DataFrame(
        {
            "id": [9001, 9002, 9003, 9004],
            "a": [1.1, 2.2, np.nan, 3.3],
            "b": [2.0, np.nan, 3.0, 1.0],
            "k": ["u", "v", None, "u"],
        }
    )
    source = tmp_path / "unlabelled.csv"
    frame.to_csv(source, index=False)
    batch_request = BatchRequest(
        release_dir=release["artifact_dir"],
        expected_release_id=release["release_id"],
        input_path=str(source),
        input_format="canonical",
        runtime_source_root=str(root),
        runtime_python=sys.executable,
        database=str(tmp_path / "project.db"),
        output_root=str(tmp_path / "jobs"),
        acknowledge_research_scoring=True,
        timeout_s=180,
    )
    pipeline = run_pipeline(batch_request, check_only=False)
    directory = Path(pipeline["artifact_dir"])
    pipeline_hash = digest(directory / "pipeline_manifest.json")
    common = {
        "pipeline_dir": directory,
        "expected_pipeline_id": pipeline["pipeline_id"],
        "expected_pipeline_manifest_sha256": pipeline_hash,
        "scoring_batch_id": pipeline["scoring_batch_id"],
        "target_unit": "synthetic_units",
    }
    prediction = pd.read_parquet(pipeline["predictions_path"])
    before = digest(tmp_path / "project.db")
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("monitoring/UI loaded model weights")
    )
    reference = NNReferenceConfig(
        **common,
        name="synthetic_reference",
        approved_population="Synthetic plumbing only; not a representative deployment claim",
    )
    checked = run(
        trained["engine"], "reference", reference, output_root=tmp_path / "refs", check_only=True
    )
    assert (
        checked["ready"]
        and (not (tmp_path / "refs").exists())
        and (before == digest(tmp_path / "project.db"))
    )
    ref = run(
        trained["engine"], "reference", reference, output_root=tmp_path / "refs", check_only=False
    )
    template = run(
        trained["engine"],
        "template",
        NNTemplateConfig(**common),
        output_root=tmp_path / "templates",
        check_only=False,
    )
    t = pd.read_csv(template["template_path"])
    assert t.y_true.isna().all() and t.observed_at.isna().all() and t.eligible_at.isna().all()
    config = NNMonitorConfig(
        **common, reference_id=ref["reference_id"], as_of="2026-01-03T00:00:00+00:00"
    )
    monitored = run(
        trained["engine"], "monitor", config, output_root=tmp_path / "monitoring", check_only=False
    )
    view = load_assessment(tmp_path / "project.db", monitored["monitoring_id"])
    assert view["performance"] is None and view["summary"]["model_type"] == "nn"
    assert view["summary"]["assessment_scope"]["performance_status"] == "unknown_no_labels"
    assert (
        view["summary"]["assessment_scope"]["distribution_status"]
        == "not_assessed_insufficient_rows"
    )
    assert view["summary"]["target_unit"] == "synthetic_units"
    labels = pd.DataFrame(
        {
            "id": prediction.id,
            "scoring_batch_id": pipeline["scoring_batch_id"],
            "y_true": prediction.y_pred_release.to_numpy() + np.array([0.0, 1.0, -1.0, 2.0]),
            "observed_at": "2026-01-02T00:00:00+00:00",
            "eligible_at": "2026-01-01T00:00:00+00:00",
        }
    )
    label_path = tmp_path / "synthetic_labels.csv"
    labels.to_csv(label_path, index=False)
    request = NNLabelConfig(
        **common,
        input_path=label_path,
        source="Synthetic test only",
        outcome_definition="Synthetic residual arithmetic",
        sampling="all_rows",
    )
    imported = run(
        trained["engine"], "labels", request, output_root=tmp_path / "labels", check_only=False
    )
    labelled = NNMonitorConfig(
        **common,
        reference_id=ref["reference_id"],
        label_import_id=imported["label_import_id"],
        as_of="2026-01-03T00:00:00+00:00",
        min_rows=2,
        min_labelled=2,
        rmse_warning=1.0,
        abs_bias_warning=0.1,
    )
    assessed = run(
        trained["engine"],
        "monitor",
        labelled,
        output_root=tmp_path / "labelled_monitoring",
        check_only=False,
    )
    saved = load_assessment(tmp_path / "project.db", assessed["monitoring_id"])
    perf = saved["performance"]
    assert (
        saved["summary"]["assessment_scope"]["performance_status"]
        == "assessed_mature_labelled_subset"
    )
    assert perf["score_metrics"]["y_pred_release"]["rmse"] == pytest.approx(np.sqrt(1.5))
    assert perf["score_metrics"]["y_pred_release"]["bias"] == pytest.approx(-0.5)
    assert any(r["metric"] == "bias" for r in metric_rows(perf))
    if corrected:
        assert perf["interval_metrics"]
    payload = monitoring_payload(tmp_path / "project.db", assessed["monitoring_id"])
    assert (
        len(payload) == 10
        and "unavailable" not in payload[0].lower()
        and ("NN assessment scope" in payload[8])
    )
    assert bundle_bytes(saved, assessed["monitoring_id"])[:2] == b"PK"
    partial = labels.copy()
    partial.loc[0, "eligible_at"] = "2026-01-10T00:00:00+00:00"
    partial.loc[1, "observed_at"] = "2026-01-10T00:00:00+00:00"
    partial.loc[2, "y_true"] = np.nan
    partial.loc[2, "observed_at"] = None
    partial_path = tmp_path / "synthetic_partial.csv"
    partial.to_csv(partial_path, index=False)
    revised_labels = NNLabelConfig(
        **common,
        input_path=partial_path,
        source="Synthetic partial only",
        outcome_definition="Synthetic residual arithmetic",
        sampling="selected",
        supersedes_label_import_id=imported["label_import_id"],
    )
    partial_id = run(
        trained["engine"],
        "labels",
        revised_labels,
        output_root=tmp_path / "partial_labels",
        check_only=False,
    )["label_import_id"]
    partial_config = NNMonitorConfig(
        **common,
        reference_id=ref["reference_id"],
        label_import_id=partial_id,
        as_of="2026-01-03T00:00:00+00:00",
        min_rows=2,
        min_labelled=2,
        rmse_warning=0.0,
        abs_bias_warning=0.0,
    )
    partial_mon = run(
        trained["engine"],
        "monitor",
        partial_config,
        output_root=tmp_path / "partial_monitoring",
        check_only=False,
    )
    partial_view = load_assessment(tmp_path / "project.db", partial_mon["monitoring_id"])
    assert partial_view["performance"]["n_evaluated"] == 1
    assert partial_view["performance"]["mature_label_coverage"] == pytest.approx(1 / 3)
    assert (
        partial_view["summary"]["assessment_scope"]["performance_status"]
        == "not_assessed_insufficient_outcomes"
    )
    assert not any(
        f["rule"] in {"rmse_warning", "mae_warning", "abs_bias_warning", "interval_undercoverage"}
        for f in partial_view["findings"]
    )
    second = run_pipeline(batch_request, check_only=False)
    second_common = {
        **common,
        "pipeline_dir": Path(second["artifact_dir"]),
        "expected_pipeline_id": second["pipeline_id"],
        "expected_pipeline_manifest_sha256": digest(
            Path(second["artifact_dir"]) / "pipeline_manifest.json"
        ),
        "scoring_batch_id": second["scoring_batch_id"],
    }
    second_monitor = NNMonitorConfig(
        **second_common, reference_id=ref["reference_id"], as_of="2026-01-03T00:00:00+00:00"
    )
    assert run(
        trained["engine"],
        "monitor",
        second_monitor,
        output_root=tmp_path / "second_monitoring",
        check_only=True,
    )["ready"]
    bad = reference.model_copy(update={"expected_pipeline_manifest_sha256": "f" * 64})
    with pytest.raises(ValueError):
        run(trained["engine"], "reference", bad, output_root=tmp_path / "bad_refs", check_only=True)
    assert not (tmp_path / "bad_refs").exists()
    batch = load_batch(trained["engine"], pipeline["scoring_batch_id"])
    wrong = reference.model_copy(update={"target_unit": "wrong"})
    with pytest.raises(ValueError):
        context_for(batch, wrong)
    assert pipeline_hash == digest(directory / "pipeline_manifest.json")
