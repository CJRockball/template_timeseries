"""Synthetic workflow; upstream resolution, reliability helper and Parquet writer are mocked."""

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.db import evaluation_registry as reg
from ml_template.evaluation import evaluate_run as ev
from ml_template.evaluation.evaluation_artifacts import verify
from ml_template.evaluation.evaluation_config import EvaluationConfig


@pytest.fixture
def world(tmp_path, monkeypatch):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE model_runs(run_id TEXT PRIMARY KEY,holdout_evaluated_at TEXT)")
    c.execute("INSERT INTO model_runs VALUES ('p',NULL)")
    c.commit()
    assignment = pd.DataFrame({"id": [1, 2, 3, 4], "partition": ["holdout"] * 4, "fold": [-1] * 4})
    inputs = SimpleNamespace(
        X=pd.DataFrame({"f": [0.1, 0.2, 0.8, 0.9]}, index=[1, 2, 3, 4]),
        y=pd.Series([1, 1, 0, 0], index=[4, 3, 2, 1]),
        assignments=assignment,
        split_id="split",
        contract=SimpleNamespace(entity_id="id", target="target"),
    )
    calls = []

    class Calibrator:
        def transform(self, p):
            return np.asarray(p) * 0.8

    class Model:
        def predict_proba(self, X):
            calls.append("predict")
            return X.f.to_numpy()

    def prepare(engine, cfg, **kw):
        cfg = EvaluationConfig.model_validate(
            {
                **cfg.model_dump(),
                "expected_silver_dataset_version_id": "s",
                "expected_split_id": "split",
            }
        )
        return SimpleNamespace(
            spec=cfg,
            identity_hash=cfg.digest(),
            fingerprint="f",
            hold_keys=[1, 2, 3, 4],
            key_col="id",
            target_col="target",
            assignments=assignment,
            feature_names=["f"],
            provenance={},
            calibrator=Calibrator() if cfg.calibration_id else None,
            calibration_dir=None,
        )

    def load(p):
        calls.append("labels")
        return inputs

    monkeypatch.setattr(ev, "prepare_evaluation", prepare)
    monkeypatch.setattr(ev, "exposure_warnings", lambda engine, p: reg.prior_exposures(engine, "s"))
    monkeypatch.setattr(ev, "_load_inputs", load)
    monkeypatch.setattr(ev, "_load_model", lambda p: Model())
    monkeypatch.setattr(
        ev,
        "_save_predictions",
        lambda frame, path: path.write_text(frame.to_json(orient="records")),
    )
    monkeypatch.setattr(ev, "reliability_rows", lambda *args: [])
    yield c, tmp_path, calls, inputs
    c.close()


def spec(**kw):
    return EvaluationConfig(parent_run_id="p", uncertainty={"enabled": False}, **kw)


def test_preflight_no_label_access(world):
    c, tmp, calls, _ = world
    assert (
        ev.evaluate_spec(c, spec(), check_only=True, output_root=tmp)["holdout_labels_loaded"]
        is False
    )
    assert calls == [] and reg.query(c, "SELECT * FROM final_evaluations") == []


def test_calibrated_evaluation_cache(world):
    c, tmp, calls, _ = world
    cfg = spec(calibration_id="cal", release_variant="calibrated")
    first = ev.evaluate_spec(c, cfg, output_root=tmp)
    assert first["summary"]["primary_value"] == 1
    assert {r["variant"] for r in verify(first["artifact_dir"])["metrics"]} == {"raw", "calibrated"}
    again = ev.evaluate_spec(c, cfg, output_root=tmp)
    assert again["cached"] and first["evaluation_id"] == again["evaluation_id"]
    assert calls == ["labels", "predict"]


def test_changed_recipe_requires_ack(world):
    c, tmp, _, _ = world
    ev.evaluate_spec(c, spec(), output_root=tmp)
    with pytest.raises(ValueError, match="previously exposed"):
        ev.evaluate_spec(c, spec(calibration_id="cal"), output_root=tmp)


def test_failure_after_access(world):
    c, tmp, _, inputs = world
    inputs.y = pd.Series([None, 0, 1, 1], index=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="binary"):
        ev.evaluate_spec(c, spec(), output_root=tmp)
    row = reg.query(c, "SELECT * FROM final_evaluations")[0]
    assert row["status"] == "failed" and row["accessed_at"]
    assert reg.query(c, "SELECT * FROM final_evaluation_metrics") == []


def test_unknown_criterion_before_access(world):
    c, tmp, calls, _ = world
    with pytest.raises(ValueError, match="unknown"):
        ev.evaluate_spec(
            c, spec(criteria=[{"metric": "typo", "operator": ">=", "value": 0}]), output_root=tmp
        )
    assert calls == []
