"""Synthetic native integration; no production dataset or holdout is used."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
@pytest.mark.parametrize("correction", ["none", "bias", "affine"])
def test_reserved_calibration_release_scoring_and_shap(
    tmp_path, monkeypatch, engineered, predictor_kind, correction
):
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    import ml_template
    from ml_template.scoring import batch as scoring
    from ml_template.training.run_config import RunConfig

    original_validate, original_score, original_read = (
        RunConfig.model_validate,
        scoring.score_batch,
        pd.read_parquet,
    )
    captured = []

    def reserve_roles(cls, raw, **kwargs):
        if isinstance(raw, dict) and raw.get("experiment_name") == "stateless_test":
            raw = dict(raw)
            raw["split"] = {
                **raw["split"],
                "calibration_fraction": 0.25,
                "recalibration_fraction": 0.15,
            }
        return original_validate(raw, **kwargs)

    def capture(engine, *args, **kwargs):
        result = original_score(engine, *args, **kwargs)
        if result.get("status") == "succeeded":
            captured.append(result)
        return result

    monkeypatch.setattr(RunConfig, "model_validate", classmethod(reserve_roles))
    monkeypatch.setattr(scoring, "score_batch", capture)
    source = Path(__file__).with_name("test_fitted_regression_pipeline_integration.py")
    spec = importlib.util.spec_from_file_location("postprocessing_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The existing fixture evaluates only synthetic data. A second synthetic evaluation
    # below explicitly acknowledges that exposure; this is not an operational recipe.
    module.test_fitted_regression_end_to_end(tmp_path, monkeypatch, engineered, predictor_kind)
    monkeypatch.setattr(pd, "read_parquet", original_read)
    (tmp_path / "features_offline").rename(tmp_path / "features")
    scored = captured[-1]
    summary = json.loads((Path(scored["artifact_dir"]) / "summary.json").read_text())
    run_id = summary["parent_run_id"]
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    assignments = pd.read_parquet(next((tmp_path / "splits").glob("*/assignments.parquet")))
    allowed = {
        role: set(assignments.loc[assignments.partition.eq(role), "id"])
        for role in ("dev", "holdout", "calibration", "recalibration")
    }
    phase = {"value": "calibration"}
    seen = []
    silver = tmp_path / "silver/test/silver_test/data.parquet"

    def partition_spy(path, *args, **kwargs):
        if Path(path) == silver and "y" in (kwargs.get("columns") or []):
            keys = set(kwargs["filters"][0][2])
            if phase["value"] == "calibration":
                assert keys in [allowed["calibration"], allowed["recalibration"]]
                assert not keys & (allowed["dev"] | allowed["holdout"])
                seen.append(keys)
            elif phase["value"] == "evaluation":
                assert keys == allowed["holdout"]
            else:
                raise AssertionError("target read during packaging or operational use")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", partition_spy)
    from ml_template.training.regression_calibration import CalibrationConfig, calibrate

    request = CalibrationConfig(
        parent_run_id=run_id,
        predictor_kind=predictor_kind,
        correction=correction,
        coverages=[0.8, 0.9, 0.95],
        target_unit="synthetic_units",
        intended_use="Synthetic test",
        acknowledge_prior_exposure=True,
        prior_exposure_note="Only synthetic fixture data",
    )
    before = len(seen)
    assert calibrate(engine, request, check_only=True)["ready"] and len(seen) == before
    fitted = calibrate(engine, request, check_only=False, output_root=tmp_path / "calibrations")
    assert len(seen) == (1 if correction == "none" else 2)
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig
    from ml_template.scoring.release_builder import build_release

    evaluation = RegressionEvaluationConfig(
        parent_run_id=run_id,
        predictor={"kind": predictor_kind},
        calibration_id=fitted["calibration_id"],
        target_unit="synthetic_units",
        release_variant="raw" if correction == "none" else "recalibrated",
        uncertainty={"enabled": True, "coverages": [0.9]},
        acknowledge_prior_exposure=True,
        prior_exposure_note="Synthetic acceptance only",
    )
    assert evaluate_spec(engine, evaluation, check_only=True)["ready"]
    phase["value"] = "evaluation"
    evaluated = evaluate_spec(engine, evaluation, output_root=tmp_path / "post_evaluations")
    phase["value"] = "package"
    release_request = RegressionReleaseConfig(
        name="postprocessed_synthetic",
        parent_run_id=run_id,
        evaluation_id=evaluated["evaluation_id"],
        calibration_id=fitted["calibration_id"],
        intended_use="Synthetic test",
        acknowledge_research_release=True,
    )
    released = build_release(engine, release_request, output_root=tmp_path / "post_releases")
    predictor = ReleasedPredictor(released["artifact_dir"])
    new = original_read(tmp_path / "unlabelled.parquet")
    whole = predictor.predict(new).set_index("id").sort_index()
    reverse = predictor.predict(new.iloc[::-1], chunk_size=1).set_index("id").sort_index()
    columns = ["y_pred_raw", "y_pred_release", "pi_lower_c900000", "pi_upper_c900000"]
    np.testing.assert_allclose(whole[columns], reverse[columns], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        (whole.pi_lower_c900000 + whole.pi_upper_c900000) / 2, whole.y_pred_release
    )
    scored = original_score(
        engine,
        released["artifact_dir"],
        tmp_path / "unlabelled.parquet",
        output_root=tmp_path / "post_scores",
        chunk_size=1,
    )
    from ml_template.monitoring.monitor_inputs import load_batch

    with sqlite3.connect(tmp_path / "project.db") as conn:
        loaded = load_batch(conn, scored["scoring_batch_id"])
    assert loaded.summary["target_unit"] == "synthetic_units"
    from ml_template.evaluation.regression_postprocessing import add_monitoring
    from ml_template.monitoring.regression import evaluate_predictions

    labels = pd.DataFrame(
        {
            "id": new.id,
            "y_true": 100000.0,
            "eligible_at": pd.Timestamp("2026-01-01", tz="UTC"),
            "observed_at": pd.Timestamp("2026-01-02", tz="UTC"),
        }
    )
    config = SimpleNamespace(
        as_of=pd.Timestamp("2026-01-03", tz="UTC"),
        min_labelled=1,
        min_label_coverage=0.9,
        target_unit="synthetic_units",
        interval_coverage_slack=0.05,
    )
    perf = evaluate_predictions(
        loaded.predictions, labels, "id", as_of=config.as_of, min_labelled=1
    )
    findings = []
    add_monitoring(
        config, loaded, {"frame": labels, "metadata": {"sampling": "all_rows"}}, perf, findings
    )
    assert perf["interval_metrics"]["c900000"]["empirical_coverage"] == 0
    assert any(item["rule"] == "interval_undercoverage" for item in findings)
    from ml_template.explainability.explain_batch import explain_batch
    from ml_template.explainability.explanation_config import ExplanationConfig
    from ml_template.explainability.presentation_data import load_explanation

    explanation = ExplanationConfig(
        task="regression",
        target_unit="synthetic_units",
        release_dir=released["artifact_dir"],
        scoring_dir=scored["artifact_dir"],
        expected_release_id=released["release_id"],
        expected_scoring_batch_id=scored["scoring_batch_id"],
        runtime_python=sys.executable,
        runtime_source_root=str(Path(ml_template.__file__).parent),
        registry_path=str(tmp_path / "project.db"),
        output_root=str(tmp_path / "explanations"),
        chunk_size=1,
    ).resolved(tmp_path)
    explained = explain_batch(explanation)
    view = load_explanation(tmp_path / "project.db", explained["explanation_id"])
    np.testing.assert_allclose(
        view.rows.base_value + view.contributions.sum(axis=1),
        view.rows.y_pred_release,
        atol=1e-5,
        rtol=1e-5,
    )
    assert view.summary["explained_prediction_column"] == "y_pred_explained"
    sidecar = Path(released["artifact_dir"]) / "regression_postprocessing.json"
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(ValueError):
        ReleasedPredictor(released["artifact_dir"])
