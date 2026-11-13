"""Real CPU NN train -> metadata -> verified predictor acceptance; synthetic data only."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=[False, True], ids=["raw", "stateless"])
def trained(request, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from ml_template import paths
    from ml_template.features.feature_set_store import build_block, create_feature_set
    from ml_template.scoring.release_artifacts import sha256
    from ml_template.training import experiment_runner as runner
    from ml_template.training.run_config import RunConfig
    from ml_template.training.split import make_split, save_split

    sr, sp, fr = tmp_path / "silver", tmp_path / "splits", tmp_path / "features"
    fr.mkdir()
    monkeypatch.setattr(paths, "DATA_SILVER", sr)
    monkeypatch.setattr(paths, "DATA_SPLITS", sp)
    monkeypatch.setattr(paths, "DATA_FEATURES", fr)
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: y}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [a, b, k], target: [y]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, a: a, b: b, k: k, y: y}\n"
        "  canonical_dtypes: {id: int64, a: float64, b: float64, k: string, y: float64}\n"
        "  value_mappings: {}\n  allowed_values: {}\n  missingness: {default_max_rate: 1.0}\n"
    )
    a = np.linspace(-3.0, 4.0, 80)
    df = pd.DataFrame(
        {
            "id": np.arange(80),
            "a": a,
            "b": np.tile([1.0, 2.0, 3.0, 2.0], 20),
            "k": np.tile(["u", "v"], 40),
        }
    )
    df["y"] = 2.5 * a + (df.k == "u").astype(float) * 0.5
    df.loc[df.id % 7 == 0, "a"] = np.nan
    df.loc[df.id % 11 == 0, "k"] = None
    sd = sr / "test/silver_test"
    sd.mkdir(parents=True)
    df.to_parquet(sd / "data.parquet", index=False)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "validation_passed": True,
                "config_sha256": sha256(contract),
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "roles": {"id": ["id"], "feature": ["a", "b", "k"], "target": ["y"]},
                "parquet_sha256": sha256(sd / "data.parquet"),
            }
        )
    )
    fsid = None
    if request.param:
        raw = df.set_index("id")[["a", "b", "k"]]
        block = build_block(
            "generic_interactions",
            raw,
            "silver_test",
            params={
                "numeric_pairs": [{"a": "a", "b": "b", "ops": ["ratio"]}],
                "category_pairs": [],
                "nullable_outputs": ["a__div__b"],
            },
            root=fr,
        )
        fs = create_feature_set(
            "silver_test",
            [block["block_id"]],
            raw_columns=list(raw),
            columns=["a", "b", "k", "a__div__b"],
            root=fr,
        )
        fsid = fs["feature_set_id"]
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "nn_package_01",
            "data": {
                "dataset_name": "test",
                "dataset_config": str(contract),
                "silver_dataset_version_id": "silver_test",
                "feature_set_id": fsid,
            },
            "split": {
                "method": "random",
                "n_folds": 2,
                "calibration_fraction": 0.1,
                "recalibration_fraction": 0.1,
            },
            "analysis": {
                "permutation_importance": False,
                "error_sample_size": 0,
                "learning_curves": False,
            },
            "model": {
                "type": "nn",
                "params": {
                    "task": "regression",
                    "hidden_dims": [4],
                    "max_epochs": 2,
                    "patience": 2,
                    "batch_size": 8,
                    "prediction_batch_size": 8,
                    "dropout": 0.0,
                    "embedding_dropout": 0.0,
                    "device": "cpu",
                    "inference_device": "cpu",
                },
            },
        }
    )
    assignment = make_split(df, cfg.split, "id", "y")
    save_split(assignment, "silver_test", cfg.split, "id", "y", df.y, sp)
    original_loader = runner.load_run_inputs
    monkeypatch.setattr(
        runner,
        "load_run_inputs",
        lambda cfg, **kwargs: original_loader(cfg, feature_root=fr, **kwargs),
    )
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    result = runner.run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=sr,
        split_root=sp,
    )
    assert not result["failures"], result
    rid = result["run_ids"][0]
    # Preserve all training evidence while testing predictor-only operations.
    parent = tmp_path / "runs" / rid
    before = {p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()}
    yield {
        "cfg": cfg,
        "engine": engine,
        "rid": rid,
        "root": parent,
        "df": df,
        "assignment": assignment,
        "roots": {"silver_root": sr, "split_root": sp, "feature_root": fr},
        "parquet": sd / "data.parquet",
        "torch": torch,
        "sha256": sha256,
        "before": before,
    }
    assert before == {
        p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()
    }
    engine.dispose()


def prepare(trained, kind):
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_nn_regression_predictor_inputs

    return prepare_nn_regression_predictor_inputs(
        trained["engine"],
        RegressionEvaluationConfig(parent_run_id=trained["rid"], predictor={"kind": kind}),
        **trained["roots"],
    )


def estimator_input(prepared, trained):
    from ml_template.features.prepared_recipe import prepare_frame

    plan = prepared.provenance["feature_preparation"]
    raw = trained["df"].set_index("id").loc[[1, 2, 3, 4], plan["recipe"]["raw_columns"]].copy()
    return prepare_frame(raw, plan["recipe"], plan["model_schema"])


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
def test_real_cpu_predictors_and_frozen_gold(trained, monkeypatch, kind):
    from ml_template.models.model_factory import load_adapter
    from ml_template.models.nn_regressor import NNRegressionAdapter
    from ml_template.models.predictor_selection import load_selected_predictor
    from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

    old_read = pd.read_parquet

    def no_source_values(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            pytest.fail("metadata/predictor accessed historical source values")
        return old_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", no_source_values)
    original_torch_load = trained["torch"].load
    monkeypatch.setattr(
        trained["torch"],
        "load",
        lambda *a, **k: pytest.fail("metadata check deserialized a checkpoint"),
    )
    p = prepare(trained, kind)
    assert p.purpose == "predictor_only" and "torch" in p.provenance["versions"]
    d = p.provenance["predictor"]
    assert d["schema_version"] == 3 and d["serialization"]["preprocessing"] == "embedded"
    monkeypatch.setattr(trained["torch"], "load", original_torch_load)
    monkeypatch.setattr(
        NNPreprocessor, "fit", lambda *a, **k: pytest.fail("prediction fitted preprocessing")
    )
    monkeypatch.setattr(
        NNRegressionAdapter, "_train", lambda *a, **k: pytest.fail("prediction fitted model")
    )
    predictor = load_selected_predictor(p.run_cfg, p.run_dir, d)
    verified = predictor.verify_checkpoints()
    assert not verified["predictions_computed"] and not verified["labels_loaded"]
    X = estimator_input(p, trained)
    whole = predictor.predict(X)
    chunks = np.concatenate([predictor.predict(X.iloc[i : i + 1]) for i in range(len(X))])
    shuffled = predictor.predict(X.iloc[::-1])[::-1]
    assert whole.shape == (len(X),) and np.isfinite(whole).all()
    np.testing.assert_allclose(whole, chunks, rtol=0, atol=1e-6)
    np.testing.assert_allclose(whole, shuffled, rtol=0, atol=1e-6)
    direct = []
    dev = trained["assignment"].query("partition == 'dev'")
    for m in d["members"]:
        adapter = load_adapter(p.run_cfg, p.run_dir / m["model_path"])
        gold = adapter.preprocessor.state_dict()
        keys = dev.id if m["fold_id"] is None else dev.loc[dev.fold != m["fold_id"], "id"]
        assert gold["fit_rows"] == len(keys) == d["training_rows"][m["model_path"]]
        observed = trained["df"].set_index("id").loc[keys, "a"].dropna()
        assert gold["numeric_state"]["a"]["fill"] == pytest.approx(float(observed.median()))
        assert gold["numeric_state"]["a"]["center"] == pytest.approx(float(observed.mean()))
        direct.append(adapter.predict(X))
        assert adapter.preprocessor.state_dict() == gold
    np.testing.assert_allclose(whole, np.mean(direct, axis=0), rtol=0, atol=1e-6)
    missing = X.copy()
    missing.loc[missing.index[0], "b"] = np.nan
    unknown = X.copy()
    unknown["k"] = unknown["k"].astype(object)
    unknown.loc[unknown.index[0], "k"] = "unseen"
    assert np.isfinite(predictor.predict(missing)).all()
    assert np.isfinite(predictor.predict(unknown)).all()
    with pytest.raises(ValueError):
        predictor.predict(X.assign(y=0.0))
    with pytest.raises(ValueError):
        predictor.predict(X.set_axis([1, 1, 2, 3]))


def test_lifecycle_gate_still_blocks_nn(trained, monkeypatch):
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import load_holdout, prepare_regression

    old = pd.read_parquet

    def forbid_source(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            pytest.fail("unauthorised source decode")
        return old(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", forbid_source)
    spec = RegressionEvaluationConfig(parent_run_id=trained["rid"])
    with pytest.raises(ValueError, match="Expected XGB"):
        prepare_regression(trained["engine"], spec, **trained["roots"])
    p = prepare(trained, "final")
    with pytest.raises(ValueError, match="does not authorise"):
        load_holdout(p)


def test_wrong_config_and_gold_partition_binding(trained):
    from ml_template.models.model_factory import load_adapter
    from ml_template.models.nn_regression_predictor import validate_gold
    from ml_template.models.predictor_selection import load_selected_predictor

    p = prepare(trained, "final")
    d = p.provenance["predictor"]
    wrong = p.run_cfg.model_copy(update={"seed": p.run_cfg.seed + 1})
    with pytest.raises(ValueError, match="snapshot differs"):
        load_selected_predictor(wrong, p.run_dir, d)
    m = d["members"][0]
    adapter = load_adapter(p.run_cfg, p.run_dir / m["model_path"])
    changed = copy.deepcopy(d)
    changed["training_rows"][m["model_path"]] += 1
    with pytest.raises(ValueError, match="fit-row count"):
        validate_gold(p.run_cfg, p.feature_info, changed, m, adapter)


def test_checkpoint_hash_checked_before_torch_load(trained, tmp_path, monkeypatch):
    import shutil

    from ml_template.models.predictor_selection import load_selected_predictor

    p = prepare(trained, "final")
    clone = tmp_path / "tampered_clone"
    shutil.copytree(p.run_dir, clone)
    (clone / "model.ubj").write_bytes(b"not an authenticated checkpoint")
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("tampered checkpoint deserialized")
    )
    with pytest.raises(ValueError, match="changed"):
        load_selected_predictor(p.run_cfg, clone, p.provenance["predictor"])


@pytest.mark.parametrize("kind", ["task", "seed", "preprocessing"])
def test_invalid_payload_semantics(trained, tmp_path, kind):
    from ml_template.models.nn_regression_predictor import validate_gold
    from ml_template.models.nn_regressor import NNRegressionAdapter

    p = prepare(trained, "final")
    d = p.provenance["predictor"]
    raw = trained["torch"].load(p.run_dir / "model.ubj", map_location="cpu", weights_only=True)
    if kind == "task":
        raw["task"] = "binary_classification"
    elif kind == "seed":
        raw["seed"] += 1
    else:
        raw["preprocessor"]["config"]["numeric_strategy"] = "mean"
    # New standalone invalid fixture; never edit/reseal the trained parent.
    bad = tmp_path / ("bad_" + kind + ".ubj")
    trained["torch"].save(raw, bad)
    with pytest.raises(ValueError):
        adapter = NNRegressionAdapter.load(p.run_cfg.model.params, p.run_cfg.seed, bad)
        validate_gold(p.run_cfg, p.feature_info, d, d["members"][0], adapter)
