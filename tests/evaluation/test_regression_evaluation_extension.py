"""Continuous predictors plus label-free, immutable regression evaluation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.regression_config import RegressionEvaluationConfig
from ml_template.models import predictor_selection as selection
from ml_template.models.regression_predictor import continuous


def predictor_world(tmp_path, kind="cv_mean"):
    cfg = SimpleNamespace(task="regression", fitted_features=None, split=SimpleNamespace(n_folds=2))
    (tmp_path / "fold_models").mkdir(exist_ok=True)
    (tmp_path / "model.ubj").write_text("final")
    for fold in [0, 1]:
        (tmp_path / "fold_models" / f"fold_{fold}.ubj").write_text(str(fold))
    pd.DataFrame({"fold": [0, 1]}).to_csv(tmp_path / "cv_folds.csv", index=False)
    expected = {
        str(p.relative_to(tmp_path)): selection.file_sha256(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }

    def verify(root, name):
        if selection.file_sha256(Path(root) / name) != expected[name]:
            raise ValueError("original artifact differs")

    descriptor = selection.describe_predictor(
        cfg, tmp_path, kind, fold_ids=[0, 1], verify_artifact=verify
    )
    return cfg, descriptor


def test_continuous_final_and_fold_mean(tmp_path):
    cfg, descriptor = predictor_world(tmp_path)
    X = pd.DataFrame({"x": [-2.0, 6.0]}, index=[10, 20])
    seen = []

    def loader(cfg, path):
        seen.append(path.name)
        offset = int(path.stem.split("_")[-1]) if path.stem.startswith("fold_") else 0

        class Adapter:
            def predict(self, frame):
                result = frame.x.to_numpy() + 2 * offset
                frame.iloc[0, 0] = 99999
                return result

        return Adapter()

    model = selection.load_selected_predictor(cfg, tmp_path, descriptor, adapter_loader=loader)
    np.testing.assert_allclose(model.predict(X), [-1.0, 7.0])
    assert seen == ["fold_0.ubj", "fold_1.ubj"]
    assert X.x.tolist() == [-2.0, 6.0]
    final = selection.describe_predictor(
        cfg, tmp_path, "final", verify_artifact=lambda root, name: None
    )
    assert final["schema_version"] == 2 and final["prediction_kind"] == "continuous"
    np.testing.assert_allclose(
        selection.load_selected_predictor(cfg, tmp_path, final, adapter_loader=loader).predict(X),
        X.x,
    )


@pytest.mark.parametrize("values", [[True, False], [1.0, np.inf], [[1.0], [2.0]], ["a", "b"]])
def test_invalid_prediction_vectors(values):
    with pytest.raises(ValueError):
        continuous(values, 2)


def test_task_descriptor_binding_and_tampering(tmp_path):
    cfg, descriptor = predictor_world(tmp_path)
    legacy = {k: v for k, v in descriptor.items() if k not in {"task", "prediction_kind"}}
    legacy["schema_version"] = 1
    with pytest.raises(ValueError):
        selection.load_selected_predictor(cfg, tmp_path, legacy)
    changed = copy.deepcopy(descriptor)
    changed["members"][0]["weight"] = 0.6
    with pytest.raises(ValueError):
        selection.required_predictor_files(changed)
    (tmp_path / "fold_models/fold_1.ubj").write_text("tampered")
    with pytest.raises(ValueError):
        selection.verify_predictor_files(tmp_path, descriptor)


def test_incomplete_fold_inventory_rejected(tmp_path):
    cfg, descriptor = predictor_world(tmp_path)
    with pytest.raises(ValueError):
        selection.describe_predictor(
            cfg, tmp_path, "cv_mean", fold_ids=[0], verify_artifact=lambda root, name: None
        )
    (tmp_path / "fold_models/fold_1.ubj").unlink()
    with pytest.raises(ValueError):
        selection.verify_predictor_files(tmp_path, descriptor)


@pytest.mark.parametrize(
    "extra",
    [
        {"policy": {"kind": "threshold", "value": 0.5}},
        {"calibration_id": "cal_1"},
        {"uncertainty": {"enabled": True}},
        {
            "criteria": [
                {"metric": "rmse", "operator": "<=", "value": 5.0, "basis": "confidence_bound"}
            ]
        },
    ],
)
def test_unsupported_evaluation_options_rejected(extra):
    with pytest.raises(ValueError):
        RegressionEvaluationConfig(parent_run_id="run_1", **extra)


@pytest.fixture
def world(tmp_path):
    pytest.importorskip("pyarrow")
    sa = pytest.importorskip("sqlalchemy")
    from ml_template.tracking.run_artifacts import sha256_file
    from ml_template.training.experiment_runner import run_experiment
    from ml_template.training.run_config import RunConfig
    from ml_template.training.split import make_split, save_split

    df = pd.DataFrame({"id": np.arange(120), "x": np.linspace(-5.0, 20.0, 120)})
    df["strength"] = 2 * df.x + 0.25
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: strength}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [x], target: [strength]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, x: x, strength: strength}\n"
    )
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "eval_test",
            "data": {
                "dataset_name": "test",
                "dataset_config": str(contract),
                "silver_dataset_version_id": "silver_test",
            },
            "split": {"method": "random", "n_folds": 3},
            "model": {
                "type": "xgb",
                "params": {"objective": "reg:squarederror", "n_estimators": 20},
            },
        }
    )
    sr, sp = tmp_path / "silver", tmp_path / "splits"
    sd = sr / "test/silver_test"
    sd.mkdir(parents=True)
    df.to_parquet(sd / "data.parquet", index=False)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "validation_passed": True,
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "roles": {"id": ["id"], "feature": ["x"], "target": ["strength"]},
                "parquet_sha256": sha256_file(sd / "data.parquet"),
            }
        )
    )
    assign = make_split(df, cfg.split, "id", "strength")
    save_split(assign, "silver_test", cfg.split, "id", "strength", df.strength, sp)

    class Fake:
        max_rounds = 20

        def fit_with_validation(self, Xt, yt, Xv, yv):
            return 3

        def refit(self, X, y, n_rounds):
            self.mean = float(y.mean())

        def predict(self, X):
            return np.full(len(X), self.mean)

        def save(self, path):
            path.write_text("fake-model")

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    summary = run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=sr,
        split_root=sp,
        factory_builder=lambda cfg: Fake,
    )
    assert not summary["failures"]
    run_id = summary["run_ids"][0]
    roots = {"silver_root": sr, "split_root": sp, "output_root": tmp_path / "evaluations"}

    def loader(cfg, path):
        return SimpleNamespace(predict=lambda X: 2 * X.x.to_numpy())

    return engine, run_id, roots, loader, tmp_path


def test_preflight_never_reads_labels(world, monkeypatch):
    from ml_template.evaluation.evaluate_run import evaluate_spec

    engine, rid, roots, loader, _ = world
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        if Path(path).name == "data.parquet":
            raise AssertionError("Preflight attempted to read target/feature data")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    checked = evaluate_spec(
        engine, RegressionEvaluationConfig(parent_run_id=rid), check_only=True, **roots
    )
    assert checked["ready"] and not checked["holdout_labels_loaded"]
    from ml_template.db import evaluation_registry

    assert not evaluation_registry.prior_exposures(engine, "silver_test")


def test_execute_cache_and_holdout_guard(world, monkeypatch):
    from ml_template.db import evaluation_registry as registry
    from ml_template.evaluation import regression_evaluate as re
    from ml_template.evaluation.evaluate_run import evaluate_spec

    engine, rid, roots, loader, tmp_path = world
    spec = RegressionEvaluationConfig(
        parent_run_id=rid, criteria=[{"metric": "rmse", "operator": "<=", "value": 0.5}]
    )
    first = evaluate_spec(engine, spec, adapter_loader=loader, **roots)
    assert first["summary"]["primary_value"] == pytest.approx(0.25)
    assert first["summary"]["criteria_status"] == "passed"
    directory = Path(first["artifact_dir"])
    predictions = pd.read_parquet(directory / "holdout_predictions.parquet")
    assert predictions.id.is_unique and "y_pred" in predictions
    assert "n_positives" not in first["summary"]
    assert registry.prior_exposures(engine, "silver_test")
    monkeypatch.setattr(
        re, "load_holdout", lambda p: (_ for _ in ()).throw(AssertionError("Cache reloaded labels"))
    )
    cached = evaluate_spec(engine, spec, **roots)
    assert cached["cached"] and cached["evaluation_id"] == first["evaluation_id"]
    another = spec.model_copy(
        update={"predictor": spec.predictor.model_copy(update={"kind": "cv_mean"})}
    )
    checked = evaluate_spec(engine, another, check_only=True, **roots)
    assert not checked["ready"]
    with pytest.raises(ValueError, match="previously exposed"):
        evaluate_spec(engine, another, adapter_loader=loader, **roots)
    (directory / "metrics.csv").write_text("tampered")
    with pytest.raises(ValueError):
        evaluate_spec(engine, spec, **roots)


def test_failed_prediction_keeps_exposure_and_releases_lock(world):
    from ml_template.db import evaluation_registry as registry
    from ml_template.evaluation.evaluate_run import evaluate_spec

    engine, rid, roots, _, _ = world

    def bad(cfg, path):
        raise RuntimeError("synthetic predictor load failure")

    spec = RegressionEvaluationConfig(parent_run_id=rid)
    with pytest.raises(RuntimeError, match="synthetic"):
        evaluate_spec(engine, spec, adapter_loader=bad, **roots)
    assert registry.prior_exposures(engine, "silver_test")
    assert not registry.query(engine, "SELECT * FROM final_evaluation_locks")
    assert registry.query(engine, "SELECT status FROM final_evaluations")[0]["status"] == "failed"


def test_cv_mean_evaluation_has_distinct_identity(world):
    from ml_template.evaluation.evaluate_run import evaluate_spec

    engine, rid, roots, loader, _ = world
    final = RegressionEvaluationConfig(parent_run_id=rid)
    mean = RegressionEvaluationConfig(parent_run_id=rid, predictor={"kind": "cv_mean"})
    a = evaluate_spec(engine, final, check_only=True, **roots)
    b = evaluate_spec(engine, mean, check_only=True, **roots)
    assert a["identity_hash"] != b["identity_hash"]
    result = evaluate_spec(engine, mean, adapter_loader=loader, **roots)
    assert result["summary"]["predictor_kind"] == "cv_mean"
    assert result["summary"]["primary_value"] == pytest.approx(0.25)


def test_preflight_rejects_tampered_training_model(world):
    from ml_template.db import evaluation_registry, model_run_registry
    from ml_template.evaluation.evaluate_run import evaluate_spec

    engine, rid, roots, loader, _ = world
    parent = model_run_registry.get_run(engine, rid)
    from ml_template.tracking.run_artifacts import resolve_stored_path

    (resolve_stored_path(parent["artifact_dir"]) / "model.ubj").write_text("changed")
    with pytest.raises(ValueError):
        evaluate_spec(
            engine, RegressionEvaluationConfig(parent_run_id=rid), check_only=True, **roots
        )
    assert not evaluation_registry.prior_exposures(engine, "silver_test")
