"""Regression training lifecycle and binary compatibility tests."""

import json

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.regression_metrics import regression_metrics
from ml_template.training.cross_validation import cross_validate, fit_final
from ml_template.training.run_config import RunConfig, XGBRegressionParams
from ml_template.training.split import make_split, save_split


def raw_config():
    return {
        "experiment_name": "reg_test",
        "task": "regression",
        "data": {
            "dataset_name": "test",
            "dataset_config": "unused.yaml",
            "silver_dataset_version_id": "silver_test",
        },
        "split": {"method": "random", "n_folds": 3},
        "model": {
            "type": "xgb",
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 30,
                "early_stopping_rounds": 3,
            },
        },
    }


def test_config_roundtrip_and_gates():
    cfg = RunConfig.model_validate(raw_config())
    assert isinstance(cfg.model.params, XGBRegressionParams)
    assert cfg.metrics.primary == "rmse" and cfg.metrics.alert_rates == []
    assert not cfg.analysis.permutation_importance
    assert RunConfig.model_validate_json(cfg.model_dump_json()) == cfg
    for extra in [
        {"split": {"method": "stratified"}},
        {"analysis": {"permutation_importance": True}},
        {"fitted_features": [{"type": "unsupported"}]},
    ]:
        with pytest.raises(ValueError):
            RunConfig.model_validate({**raw_config(), **extra})
    bad = raw_config()
    bad["model"]["params"]["scale_pos_weight"] = "auto"
    with pytest.raises(ValueError):
        RunConfig.model_validate(bad)


def test_metrics_and_constant_target():
    scores = regression_metrics([-1.0, 2.0, 5.0], [0.0, 2.0, 3.0])
    assert scores["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert scores["mae"] == 1.0
    assert "r2" not in regression_metrics([2.0, 2.0], [1.0, 3.0])
    with pytest.raises(ValueError):
        regression_metrics([1.0, np.inf], [1.0, 2.0])


def test_binary_config_hash_unchanged():
    import hashlib

    from ml_template.training.run_config import HASH_EXCLUDED_SECTIONS, config_hash

    cfg = RunConfig.model_validate({"experiment_name": "binary", "data": raw_config()["data"]})
    payload = cfg.model_dump(mode="json", exclude=HASH_EXCLUDED_SECTIONS)
    payload.pop("task")
    if payload.get("fitted_features") is None:
        payload.pop("fitted_features", None)
    if payload["split"].get("cv_seed") is None:
        payload["split"].pop("cv_seed", None)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    assert config_hash(cfg) == expected


def test_cv_holdout_exclusion_and_fractional_targets():
    df = pd.DataFrame(
        {"id": np.arange(150), "x": np.arange(150) / 10, "y": np.linspace(-3.25, 40.75, 150)}
    )
    cfg = RunConfig.model_validate(raw_config())
    assignments = make_split(df, cfg.split, "id", "y")
    X = df.set_index("id")[["x"]]
    y = df.set_index("id").y
    hold = set(assignments.loc[assignments.partition == "holdout", "id"])
    seen = []

    class Fake:
        max_rounds = 30

        def fit_with_validation(self, Xt, yt, Xv, yv):
            assert not set(Xt.index) & set(Xv.index)
            assert not (set(Xt.index) | set(Xv.index)) & hold
            assert np.any(yt.to_numpy() != yt.to_numpy().astype(int))
            seen.append((set(Xt.index), set(Xv.index)))
            return 4

        def refit(self, X, y, n_rounds):
            assert not set(X.index) & hold
            self.keys = set(X.index)
            self.value = float(y.mean())

        def predict(self, X):
            assert not set(X.index) & self.keys
            return np.full(len(X), self.value)

    cv = cross_validate(
        Fake,
        X,
        y,
        assignments,
        "id",
        validation_fraction=0.2,
        seed=42,
        metric_names=cfg.metrics.names,
        alert_rates=[],
        task="regression",
    )
    assert set(cv.oof.id) == set(X.index) - hold
    assert cv.oof.id.is_unique and "y_pred_oof" in cv.oof and "p_oof" not in cv.oof
    assert {"rmse", "mae", "r2"} <= {r["metric"] for r in cv.metric_rows}
    assert "base_rate" not in {r["metric"] for r in cv.metric_rows}
    final = fit_final(Fake, X, y, assignments, "id", cv.n_rounds_final)
    assert final.keys == set(X.index) - hold
    assert len(seen) == 3


def test_adapter_save_load_and_early_stop(tmp_path):
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template.models.xgb_regressor import XGBRegressionAdapter

    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series(10 * X.a - 2 * X.b + 0.25)
    params = XGBRegressionParams(n_estimators=30, early_stopping_rounds=3)
    adapter = XGBRegressionAdapter(params, 42)
    best = adapter.fit_with_validation(X.iloc[:80], y.iloc[:80], X.iloc[80:], -y.iloc[80:])
    assert 1 <= best <= 30
    path = tmp_path / "early.ubj"
    adapter.save(path)
    restored = XGBRegressionAdapter.load(params, 42, path)
    np.testing.assert_allclose(adapter.predict(X), restored.predict(X), rtol=0, atol=1e-7)
    adapter.refit(X, y, best)
    adapter.save(tmp_path / "fixed.ubj")
    restored = XGBRegressionAdapter.load(params, 42, tmp_path / "fixed.ubj")
    np.testing.assert_allclose(adapter.predict(X), restored.predict(X), rtol=0, atol=1e-7)
    with pytest.raises(ValueError):
        restored.predict(X[["b", "a"]])


def test_runner_integration(tmp_path):
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from ml_template.db import model_run_registry as registry
    from ml_template.tracking.run_artifacts import sha256_file
    from ml_template.training import experiment_runner as er

    df = pd.DataFrame(
        {"id": np.arange(150), "x": np.linspace(0, 1, 150), "y": np.linspace(-2.5, 77.5, 150)}
    )
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {entity_id: id, target: y, task: regression}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [x], target: [y]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, x: x, y: y}\n"
    )
    raw = raw_config()
    raw["data"]["dataset_config"] = str(contract)
    cfg = RunConfig.model_validate(raw)
    sr, sp = tmp_path / "silver", tmp_path / "splits"
    sd = sr / "test" / "silver_test"
    sd.mkdir(parents=True)
    df.to_parquet(sd / "data.parquet", index=False)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "roles": {"id": ["id"], "feature": ["x"], "target": ["y"]},
                "parquet_sha256": sha256_file(sd / "data.parquet"),
                "validation_passed": True,
            }
        )
    )
    assignments = make_split(df, cfg.split, "id", "y")
    save_split(assignments, "silver_test", cfg.split, "id", "y", df.y, sp)

    class Fake:
        max_rounds = 30

        def fit_with_validation(self, Xt, yt, Xv, yv):
            return 4

        def refit(self, X, y, n_rounds):
            self.mean = float(y.mean())

        def predict(self, X):
            return np.full(len(X), self.mean)

        def save(self, path):
            path.write_text("fake-regressor")

    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    result = er.run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=sr,
        split_root=sp,
        factory_builder=lambda cfg: Fake,
    )
    assert not result["failures"]
    run_id = result["run_ids"][0]
    record = registry.get_run(engine, run_id)
    assert record["status"] == "succeeded" and record["primary_metric"] == "rmse"
    rd = tmp_path / "runs" / run_id
    assert (rd / "task.json").exists()
    assert (rd / "model.ubj").exists()
    assert "y_pred_oof" in pd.read_parquet(rd / "oof_predictions.parquet")
    assert not (rd / "analysis_errors.json").exists()
