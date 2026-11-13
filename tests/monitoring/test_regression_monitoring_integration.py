"""Native score artifacts -> reference -> delayed outcomes -> monitoring.

Uses the existing fitted-regression synthetic pipeline acceptance fixture.
No verification boundary or model is mocked during monitoring.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
def test_native_regression_monitoring(tmp_path, monkeypatch, engineered, predictor_kind):
    pytest.importorskip("pyarrow")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template.scoring import batch as scoring

    original = scoring.score_batch
    captured = []

    def capture(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        if not kwargs.get("check_only", False) and result.get("status") == "succeeded":
            captured.append((engine, result))
        return result

    monkeypatch.setattr(scoring, "score_batch", capture)
    source = Path(__file__).parents[1] / "scoring/test_fitted_regression_pipeline_integration.py"
    spec = importlib.util.spec_from_file_location("fitted_monitoring_acceptance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_fitted_regression_end_to_end(tmp_path, monkeypatch, engineered, predictor_kind)
    assert captured
    engine, scored = captured[-1]

    from ml_template.models import model_factory
    from ml_template.monitoring.artifacts import read_json, verify
    from ml_template.monitoring.label_inputs import import_labels
    from ml_template.monitoring.monitor_batch import evaluate
    from ml_template.monitoring.monitor_config import LabelConfig, MonitorConfig, ReferenceConfig
    from ml_template.monitoring.monitor_inputs import load_batch
    from ml_template.monitoring.reference import build_reference

    def forbidden(*args, **kwargs):
        raise AssertionError("monitoring must not load or fit a model")

    monkeypatch.setattr(model_factory, "load_adapter", forbidden)
    batch = load_batch(engine, scored["scoring_batch_id"])
    root = tmp_path / "monitoring"
    ref_cfg = ReferenceConfig(
        scoring_batch_id=batch.batch_id,
        name="synthetic",
        approved_population="synthetic plumbing only",
        target_unit="synthetic_units",
    )
    assert build_reference(engine, ref_cfg, output_root=root, check_only=True)["ready"]
    assert not root.exists()
    reference = build_reference(engine, ref_cfg, output_root=root)
    predictions = batch.predictions
    labels = pd.DataFrame(
        {
            batch.key: predictions[batch.key],
            "scoring_batch_id": batch.batch_id,
            "y_true": predictions.y_pred_release + 0.25,
            "observed_at": "2026-01-01T00:00:00Z",
            "eligible_at": "2026-01-01T00:00:00Z",
        }
    )
    labels.loc[labels.index[-1], ["y_true", "observed_at"]] = [np.nan, None]
    label_path = tmp_path / "delayed_outcomes.parquet"
    labels.to_parquet(label_path, index=False)
    label_cfg = LabelConfig(
        scoring_batch_id=batch.batch_id,
        input_path=label_path,
        source="synthetic fixture",
        outcome_definition="synthetic outcome",
        sampling="all_rows",
        target_unit="synthetic_units",
    )
    imported = import_labels(engine, label_cfg, output_root=root)
    config = MonitorConfig(
        scoring_batch_id=batch.batch_id,
        reference_id=reference["reference_id"],
        label_import_id=imported["label_import_id"],
        target_unit="synthetic_units",
        as_of="2026-01-02T00:00:00Z",
        min_rows=1,
        min_labelled=1,
        min_label_coverage=0.5,
        rmse_warning=0.1,
    )
    assert evaluate(engine, config, output_root=root, check_only=True)["ready"]
    monitored = evaluate(engine, config, output_root=root)
    directory = Path(monitored["artifact_dir"])
    verify(directory)
    performance = read_json(directory / "performance.json")
    assert performance["score_metrics"]["y_pred_release"]["rmse"] == pytest.approx(0.25)
    assert performance["mature_label_coverage"] == pytest.approx((len(labels) - 1) / len(labels))
    assert read_json(directory / "summary.json")["task"] == "regression"
    rules = {item["rule"] for item in read_json(directory / "findings.json")}
    assert {"rmse_warning", "self_reference"} <= rules
    # Integrity rejection occurs before any attempt to decode this corrupted Parquet.
    path = Path(scored["artifact_dir"]) / "predictions.parquet"
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        load_batch(engine, batch.batch_id)
