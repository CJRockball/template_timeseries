"""Real synthetic Silver/split -> shared CV/refit -> registered NN checkpoints.

Does NOT claim evaluation, release, calibration or NN explanation acceptance.
"""

import json

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("engineered", [False, True])
def test_nn_regression_shared_runner(tmp_path, monkeypatch, engineered):
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from ml_template import paths
    from ml_template.features.feature_set_store import build_block, create_feature_set
    from ml_template.models.model_factory import load_adapter
    from ml_template.preprocessing.nn_preprocessor import NNPreprocessor
    from ml_template.scoring.release_artifacts import sha256
    from ml_template.training import experiment_runner as runner
    from ml_template.training.run_config import RunConfig
    from ml_template.training.run_inputs import load_run_inputs
    from ml_template.training.split import make_split, save_split

    silver, splits, features = tmp_path / "silver", tmp_path / "splits", tmp_path / "features"
    features.mkdir()
    monkeypatch.setattr(paths, "DATA_SILVER", silver)
    monkeypatch.setattr(paths, "DATA_SPLITS", splits)
    monkeypatch.setattr(paths, "DATA_FEATURES", features)
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: y}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [a, b, k], target: [y]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, a: a, b: b, k: k, y: y}\n"
        "  canonical_dtypes: {id: int64, a: float64, b: float64, k: string, y: float64}\n"
        "  value_mappings: {}\n  allowed_values: {}\n  missingness: {default_max_rate: 0.0}\n"
    )
    df = pd.DataFrame(
        {
            "id": np.arange(120),
            "a": np.linspace(-3.0, 3.0, 120),
            "b": np.tile([2.0, 3.0, 4.0], 40),
            "k": np.tile(["u", "v"], 60),
        }
    )
    df["y"] = 2 * df.a / df.b - 0.25
    destination = silver / "test/silver_test"
    destination.mkdir(parents=True)
    df.to_parquet(destination / "data.parquet", index=False)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "validation_passed": True,
                "config_sha256": sha256(contract),
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "roles": {"id": ["id"], "feature": ["a", "b", "k"], "target": ["y"]},
                "parquet_sha256": sha256(destination / "data.parquet"),
            }
        )
    )
    feature_id = None
    if engineered:
        source = df.set_index("id")[["a", "b", "k"]]
        block = build_block(
            "generic_interactions",
            source,
            "silver_test",
            params={"numeric_pairs": [{"a": "a", "b": "b", "ops": ["ratio"]}]},
            root=features,
        )
        feature_set = create_feature_set(
            "silver_test",
            [block["block_id"]],
            raw_columns=list(source),
            columns=["a__div__b", "a", "b", "k"],
            root=features,
        )
        feature_id = feature_set["feature_set_id"]
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "nn_native_training",
            "data": {
                "dataset_name": "test",
                "dataset_config": str(contract),
                "silver_dataset_version_id": "silver_test",
                "feature_set_id": feature_id,
            },
            "split": {
                "method": "random",
                "n_folds": 3,
                "calibration_fraction": 0.15,
                "recalibration_fraction": 0.1,
            },
            "model": {
                "type": "nn",
                "params": {
                    "task": "regression",
                    "hidden_dims": [8],
                    "max_epochs": 3,
                    "patience": 2,
                    "batch_size": 16,
                    "dropout": 0.0,
                    "embedding_dropout": 0.0,
                    "device": "cpu",
                },
            },
        }
    )
    assignments = make_split(df, cfg.split, "id", "y")
    save_split(assignments, "silver_test", cfg.split, "id", "y", df.y, splits)
    dev = set(assignments.loc[assignments.partition.eq("dev"), "id"])
    fitted_keys = []
    fit = NNPreprocessor.fit

    def track_fit(self, frame):
        fitted_keys.append(set(frame.index))
        assert set(frame.index) <= dev
        return fit(self, frame)

    monkeypatch.setattr(NNPreprocessor, "fit", track_fit)
    original_loader = runner.load_run_inputs
    monkeypatch.setattr(
        runner,
        "load_run_inputs",
        lambda cfg, **kwargs: original_loader(cfg, feature_root=features, **kwargs),
    )
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    trained = runner.run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=silver,
        split_root=splits,
    )
    assert not trained["failures"], trained["failures"]
    run_dir = tmp_path / "runs" / trained["run_ids"][0]
    assert fitted_keys[-1] == dev
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    assert set(oof.id) == dev and "p_oof" not in oof
    assert np.isfinite(oof.y_pred_oof).all()
    assert (run_dir / "early_stopping_curves.parquet").is_file()
    loaded = load_run_inputs(cfg, silver_root=silver, split_root=splits, feature_root=features)
    adapter = load_adapter(cfg, run_dir / "model.ubj")
    assert adapter.preprocessor.state_dict()["fit_rows"] == len(dev)
    assert adapter.preprocessor.state_dict()["numeric_state"]["a"]["center"] == pytest.approx(
        loaded.X.loc[sorted(dev), "a"].mean()
    )
    assert np.isfinite(adapter.predict(loaded.X.loc[sorted(dev)])).all()
    assert len(list((run_dir / "fold_models").glob("fold_*.ubj"))) == cfg.split.n_folds
    for fold in range(cfg.split.n_folds):
        member = load_adapter(cfg, run_dir / "fold_models" / f"fold_{fold}.ubj")
        keys = assignments.loc[assignments.partition.eq("dev") & assignments.fold.ne(fold), "id"]
        state = member.preprocessor.state_dict()
        assert state["fit_rows"] == len(keys)
        assert state["numeric_state"]["a"]["center"] == pytest.approx(
            loaded.X.loc[keys, "a"].mean()
        )
    # Downstream acceptance is deliberately NOT enabled by this training-only patch.
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_regression

    with pytest.raises(ValueError, match="Expected XGB regression parent"):
        prepare_regression(
            engine,
            RegressionEvaluationConfig(parent_run_id=trained["run_ids"][0]),
            silver_root=silver,
            split_root=splits,
            feature_root=features,
        )
