import sqlite3

import numpy as np
import pytest
from pydantic import ValidationError

from ml_template.db import evaluation_registry as reg
from ml_template.evaluation.decision_policy import apply_policy, policy_metrics, probabilities
from ml_template.evaluation.evaluation_config import EvaluationConfig, PolicyConfig
from ml_template.evaluation.evaluation_inputs import canonical_keys, holdout_fingerprint
from ml_template.evaluation.evaluation_statistics import assess_criteria, score_metrics
from ml_template.evaluation.evaluation_uncertainty import bootstrap_metrics


@pytest.mark.parametrize(
    "data",
    [
        {"release_variant": "calibrated"},
        {"unknown": 1},
        {"alert_rates": [0]},
        {"alert_rates": [0.1, 0.1]},
        {"acknowledge_prior_exposure": True},
        {"reliability_bins": 1},
        {"uncertainty": {"sampling_unit": "group"}},
    ],
)
def test_invalid_config(data):
    with pytest.raises(ValidationError):
        EvaluationConfig(parent_run_id="p", **data)


@pytest.mark.parametrize(
    "kind,value",
    [("none", 1), ("threshold", None), ("threshold", -1), ("alert_rate", 0), ("alert_rate", 2)],
)
def test_invalid_policy(kind, value):
    with pytest.raises(ValidationError):
        PolicyConfig(kind=kind, value=value)


@pytest.mark.parametrize("p", [[], [np.nan], [np.inf], [-1], [1.1], [[0.1]]])
def test_bad_probabilities(p):
    with pytest.raises(ValueError):
        probabilities(p)


def test_policy_rounding_ties_and_threshold():
    p = [0.5, 0.5, 0.5, 0.1, 0]
    assert apply_policy(p, PolicyConfig()) is None
    assert apply_policy(p, PolicyConfig(kind="alert_rate", value=0.3)).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert apply_policy(p, PolicyConfig(kind="threshold", value=0.5)).sum() == 3


def test_metrics_and_undefined():
    r = score_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [0.5])
    assert r["pr_auc"] == r["roc_auc"] == r["precision_at_50pct"] == 1
    assert r["brier"] == pytest.approx(0.025)
    assert score_metrics([0, 0], [0.1, 0.2])["pr_auc"] is None
    assert policy_metrics([0, 0], [False, False])["precision"] is None


def test_fingerprint_and_spec_hash():
    assert holdout_fingerprint("s", "id", [1, 2]) == holdout_fingerprint("s", "id", [2, 1])
    assert holdout_fingerprint("s", "id", [1]) != holdout_fingerprint("s", "id", ["1"])
    assert (
        EvaluationConfig(parent_run_id="p").digest() != EvaluationConfig(parent_run_id="q").digest()
    )


@pytest.mark.parametrize("keys", [[], [None], [1, 1], [1.5]])
def test_bad_keys(keys):
    with pytest.raises(ValueError):
        canonical_keys(keys)


def test_paired_bootstrap_reproducible():
    cfg = EvaluationConfig(parent_run_id="p", calibration_id="c", uncertainty={"repetitions": 100})
    y = np.tile([0, 0, 0, 1], 10)
    p = np.linspace(0.1, 0.9, len(y))
    a = bootstrap_metrics(y, {"raw": p, "calibrated": p}, cfg)
    assert a == bootstrap_metrics(y, {"raw": p, "calibrated": p}, cfg)
    assert all(r["value"] == r["ci_low"] == r["ci_high"] == 0 for r in a[1])


def test_criteria_and_disabled_bootstrap():
    cfg = EvaluationConfig(
        parent_run_id="p",
        uncertainty={"enabled": False},
        criteria=[{"metric": "brier", "operator": "<=", "value": 0.1}],
    )
    rows, _ = bootstrap_metrics([0, 1], {"raw": [0.1, 0.9]}, cfg)
    assert assess_criteria(rows, cfg)["status"] == "met"
    assert all(r["ci_low"] is None for r in rows)


@pytest.fixture
def db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE model_runs (run_id TEXT PRIMARY KEY, holdout_evaluated_at TEXT)")
    c.execute("INSERT INTO model_runs VALUES ('p',NULL)")
    c.commit()
    yield c
    c.close()


def start(db, eid="e1", identity="h1"):
    reg.start_evaluation(
        db,
        evaluation_id=eid,
        identity_hash=identity,
        spec=EvaluationConfig(parent_run_id="p", expected_silver_dataset_version_id="s"),
        fingerprint="f",
        artifact_dir="unused",
        prior=[],
    )


def test_registry_history_and_first_access(db):
    start(db)
    reg.mark_accessed(db, "e1", "p")
    stamp = db.execute("SELECT holdout_evaluated_at FROM model_runs").fetchone()[0]
    reg.finish_evaluation(db, "e1", rows=[{"variant": "raw", "metric": "brier", "value": 0.1}])
    assert reg.cached_evaluation(db, "h1")["evaluation_id"] == "e1"
    with pytest.raises(ValueError, match="immutable"):
        reg.finish_evaluation(db, "e1")
    start(db, "e2", "h2")
    reg.mark_accessed(db, "e2", "p")
    assert db.execute("SELECT holdout_evaluated_at FROM model_runs").fetchone()[0] == stamp
    reg.finish_evaluation(db, "e2", error="test")


def test_registry_locks_and_failure_release(db):
    start(db)
    with pytest.raises(RuntimeError, match="lock"):
        start(db, "e2", "h2")
    reg.finish_evaluation(db, "e1", error="test")
    start(db, "e2")
    assert reg.cached_evaluation(db, "h1") is None
