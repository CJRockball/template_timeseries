"""Backend boundaries; run inside the installed project environment."""

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_template.db import monitoring_registry as registry
from ml_template.monitoring.artifacts import seal, verify, write_json
from ml_template.monitoring.drift import compare, js_divergence, profile
from ml_template.monitoring.label_inputs import validate_labels
from ml_template.monitoring.monitor_config import MonitorConfig
from ml_template.monitoring.performance import evaluate_predictions


def test_monitoring_identical_distribution():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    ref = profile(values, numeric=True)
    assert compare(values.sample(frac=1, random_state=1), ref)["js_divergence"] == pytest.approx(0)


def test_monitoring_disjoint_distributions():
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)


def test_monitoring_extreme_values_are_not_dropped():
    ref = profile(pd.Series([1.0, 2.0, 3.0]), numeric=True)
    result = compare(pd.Series([-1e9, 1e9]), ref)
    assert sum(result["current_counts"]) == 2
    assert result["outside_reference_range"] == 2


def test_monitoring_unseen_reference_category():
    ref = profile(pd.Series(["L", "M"]), numeric=False)
    assert compare(pd.Series(["H", "L"]), ref)["unseen_reference_categories"] == 1


def test_monitoring_artifact_change_is_rejected(tmp_path):
    write_json(tmp_path / "summary.json", {"value": 1})
    expected = seal(tmp_path)
    verify(tmp_path, expected)
    write_json(tmp_path / "summary.json", {"value": 2})
    with pytest.raises(ValueError):
        verify(tmp_path, expected)


def test_monitoring_resealed_change_is_rejected(tmp_path):
    write_json(tmp_path / "summary.json", {"value": 1})
    expected = seal(tmp_path)
    write_json(tmp_path / "summary.json", {"value": 2})
    seal(tmp_path)
    with pytest.raises(ValueError):
        verify(tmp_path, expected)


def test_monitoring_naive_as_of_is_rejected():
    with pytest.raises(ValidationError):
        MonitorConfig(scoring_batch_id="score_a", reference_id="ref_a", as_of="2026-10-21")


def sample_data():
    predictions = pd.DataFrame(
        {
            "key": [3, 1, 2, 4],
            "p_raw": [0.8, 0.1, 0.7, 0.2],
            "p_release": [0.8, 0.1, 0.7, 0.2],
            "policy_flag": [True, False, False, False],
        }
    )
    labels = pd.DataFrame(
        {
            "key": [1, 2, 3, 4],
            "scoring_batch_id": "score_a",
            "y_true": [0, 1, 1, np.nan],
            "observed_at": ["2026-10-20T00:00:00Z"] * 3 + [None],
            "eligible_at": "2026-10-20T00:00:00Z",
        }
    )
    batch = SimpleNamespace(
        key="key", batch_id="score_a", predictions=predictions, contract={"key_kind": "int"}
    )
    return batch, labels


def test_monitoring_keyed_metrics_preserve_saved_policy():
    batch, labels = sample_data()
    result = evaluate_predictions(
        batch.predictions,
        validate_labels(labels, batch),
        batch.key,
        as_of="2026-10-21T00:00:00Z",
        min_labelled=1,
        min_positives=1,
        min_negatives=1,
    )
    assert result["n_evaluated"] == 3
    assert result["mature_label_coverage"] == 0.75
    assert result["policy"]["tp"] == 1
    assert result["policy"]["fn"] == 1
    assert result["policy"]["recall"] == 0.5


def test_monitoring_missing_labels_are_not_negative():
    batch, labels = sample_data()
    labels["y_true"] = np.nan
    labels["observed_at"] = None
    result = evaluate_predictions(
        batch.predictions,
        validate_labels(labels, batch),
        batch.key,
        as_of="2026-10-21T00:00:00Z",
        min_labelled=1,
        min_positives=1,
        min_negatives=1,
    )
    assert result["n_evaluated"] == 0
    assert result["score_metrics"] == {}


def test_monitoring_wrong_batch_labels_are_rejected():
    batch, labels = sample_data()
    labels.loc[0, "scoring_batch_id"] = "another_batch"
    with pytest.raises(ValueError):
        validate_labels(labels, batch)


def test_monitoring_completed_run_cannot_be_overwritten():
    config = MonitorConfig(
        scoring_batch_id="score_a", reference_id="ref_a", as_of="2026-10-21T00:00:00Z"
    )
    with sqlite3.connect(":memory:") as conn:
        registry.start_run(conn, "mon_a", config, "artifacts/monitoring/mon_a")
        registry.finish_run(
            conn, "mon_a", release_id="rel_a", manifest_hash="example_hash", assessment="review"
        )
        with pytest.raises(ValueError):
            registry.finish_run(conn, "mon_a", error="attempted overwrite")
