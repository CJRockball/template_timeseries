"""Predictor tests use fake saved adapters: no training, DB or holdout access."""

import copy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_template.evaluation.evaluation_config import EvaluationConfig
from ml_template.models.predictor_selection import (
    FoldMeanPredictor,
    describe_predictor,
    file_sha256,
    load_selected_predictor,
    required_predictor_files,
    safe_file,
)


@pytest.fixture
def saved_run(tmp_path):
    cfg = SimpleNamespace(fitted_features=[{"step": "mock_gold"}], split=SimpleNamespace(n_folds=3))
    (tmp_path / "fold_models").mkdir()
    pd.DataFrame({"fold": [0, 1, 2], "best_round": [2, 3, 4]}).to_csv(
        tmp_path / "cv_folds.csv", index=False
    )
    for fold, center in enumerate([0.2, 0.5, 0.8]):
        (tmp_path / "fold_models" / f"fold_{fold}.ubj").write_text(json.dumps({"fold": fold}))
        (tmp_path / "fold_models" / f"fold_{fold}.transformer.joblib").write_text(
            json.dumps({"center": center})
        )
    (tmp_path / "model.ubj").write_text("final")
    (tmp_path / "model.transformer.joblib").write_text(json.dumps({"center": 99.0}))
    original = {
        p.relative_to(tmp_path).as_posix(): file_sha256(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    verified, loaded, received = [], [], []

    def verify(root, name):
        verified.append(name)
        if name not in original or file_sha256(root / name) != original[name]:
            raise ValueError(f"original artifact manifest mismatch: {name}")

    class Adapter:
        def __init__(self, center):
            self.center = center

        def predict_proba(self, X):
            received.append((self.center, X.index.tolist()))
            values = X.x.fillna(self.center).to_numpy(dtype=float)
            result = 1 / (1 + np.exp(-(values - self.center)))
            X["x"] = -999.0  # Detect accidental sharing of a mutable fold input.
            return result

        def fit(self, *args):
            raise AssertionError("inference must never fit")

    def loader(config, path):
        loaded.append(str(path))
        transformer = path.with_name(path.stem + ".transformer.joblib")
        return Adapter(json.loads(transformer.read_text())["center"])

    descriptor = describe_predictor(
        cfg, tmp_path, "cv_mean", fold_ids=[0, 1, 2], verify_artifact=verify
    )
    return SimpleNamespace(
        root=tmp_path,
        cfg=cfg,
        descriptor=descriptor,
        verify=verify,
        loader=loader,
        loaded=loaded,
        received=received,
    )


def test_distinct_fold_gold_then_probability_average(saved_run):
    env = saved_run
    X = pd.DataFrame({"x": [0.1, np.nan, 0.9]}, index=pd.Index([30, 10, 20], name="entity"))
    original = X.copy(deep=True)
    model = FoldMeanPredictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)
    got = model.predict_proba(X)
    expected = np.mean(
        [1 / (1 + np.exp(-(X.x.fillna(c).to_numpy() - c))) for c in [0.2, 0.5, 0.8]], axis=0
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    assert [center for center, keys in env.received] == [0.2, 0.5, 0.8]
    assert all(keys == [30, 10, 20] for center, keys in env.received)
    assert all("fold_models" in path for path in env.loaded)
    pd.testing.assert_frame_equal(X, original)


def test_chunk_and_row_order_parity(saved_run):
    env = saved_run
    X = pd.DataFrame({"x": [0.1, 0.4, np.nan, 0.9]}, index=[30, 10, 40, 20])
    model = load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)
    whole = pd.Series(model.predict_proba(X), index=X.index)
    reversed_scores = pd.Series(model.predict_proba(X.iloc[::-1]), index=X.index[::-1])
    np.testing.assert_allclose(whole, reversed_scores.reindex(X.index), rtol=0, atol=0)
    chunks = np.concatenate([model.predict_proba(X.iloc[:2]), model.predict_proba(X.iloc[2:])])
    np.testing.assert_allclose(whole, chunks, rtol=0, atol=0)


def test_move_and_reload_parity(saved_run, tmp_path):
    env = saved_run
    destination = tmp_path / "moved"
    destination.mkdir()
    for name in required_predictor_files(env.descriptor):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(env.root / name, target)
    X = pd.DataFrame({"x": [0.1, 0.9]})
    before = load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)
    after = load_selected_predictor(env.cfg, destination, env.descriptor, adapter_loader=env.loader)
    np.testing.assert_array_equal(before.predict_proba(X), after.predict_proba(X))


@pytest.mark.parametrize(
    "artifact", ["fold_models/fold_0.ubj", "fold_models/fold_0.transformer.joblib"]
)
def test_tampered_member_rejected_before_loading(saved_run, artifact):
    env = saved_run
    (env.root / artifact).write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)
    assert not env.loaded


def test_member_changed_after_constructor_rejected(saved_run):
    env = saved_run
    model = load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)
    (env.root / "fold_models/fold_1.transformer.joblib").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        model.predict_proba(pd.DataFrame({"x": [0.5]}))


def test_missing_fold_is_not_silently_dropped(saved_run):
    env = saved_run
    (env.root / "fold_models/fold_2.ubj").unlink()
    with pytest.raises(ValueError):
        load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=env.loader)


def test_wrong_expected_count_rejected(saved_run):
    env = saved_run
    with pytest.raises(ValueError, match="fold count"):
        describe_predictor(
            env.cfg, env.root, "cv_mean", fold_ids=[0, 1], verify_artifact=env.verify
        )


def test_fold_inventory_disagreement_rejected(saved_run):
    env = saved_run
    pd.DataFrame({"fold": [0, 1, 9]}).to_csv(env.root / "cv_folds.csv", index=False)
    with pytest.raises(ValueError):
        describe_predictor(
            env.cfg, env.root, "cv_mean", fold_ids=[0, 1, 2], verify_artifact=env.verify
        )


def test_duplicate_fold_descriptor_rejected(saved_run):
    descriptor = copy.deepcopy(saved_run.descriptor)
    descriptor["members"][1] = copy.deepcopy(descriptor["members"][0])
    with pytest.raises(ValueError):
        required_predictor_files(descriptor)


def test_unequal_weights_rejected(saved_run):
    descriptor = copy.deepcopy(saved_run.descriptor)
    descriptor["members"][0]["weight"] = 0.5
    with pytest.raises(ValueError, match="equal"):
        required_predictor_files(descriptor)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan, 0.2]),
        np.array([-1.0, 0.2]),
        np.array([1.1, 0.2]),
        np.array([[0.2], [0.3]]),
        np.array([True, False]),
    ],
)
def test_invalid_member_probabilities_rejected(saved_run, bad):
    env = saved_run

    def loader(cfg, path):
        return SimpleNamespace(predict_proba=lambda X: bad)

    model = load_selected_predictor(env.cfg, env.root, env.descriptor, adapter_loader=loader)
    with pytest.raises(ValueError):
        model.predict_proba(pd.DataFrame({"x": [0.1, 0.2]}))


def test_final_predictor_uses_only_final_model(saved_run):
    env = saved_run
    descriptor = describe_predictor(env.cfg, env.root, "final", verify_artifact=env.verify)
    model = load_selected_predictor(env.cfg, env.root, descriptor, adapter_loader=env.loader)
    assert env.loaded == [str(env.root / "model.ubj")]
    assert required_predictor_files(descriptor) == ["model.transformer.joblib", "model.ubj"]
    assert model.center == 99.0


def test_fold_transformers_are_not_optional_when_configured(saved_run):
    env = saved_run
    descriptor = copy.deepcopy(env.descriptor)
    descriptor["members"][0]["transformer_path"] = None
    with pytest.raises(ValueError, match="transformer"):
        load_selected_predictor(env.cfg, env.root, descriptor, adapter_loader=env.loader)


def test_unfitted_preprocessing_declaration_requires_no_transformers(saved_run):
    env = saved_run
    cfg = SimpleNamespace(fitted_features=None, split=SimpleNamespace(n_folds=3))
    descriptor = describe_predictor(
        cfg, env.root, "cv_mean", fold_ids=[0, 1, 2], verify_artifact=env.verify
    )
    assert all(m["transformer_path"] is None for m in descriptor["members"])
    assert not any(name.endswith(".joblib") for name in required_predictor_files(descriptor))


def test_predictor_hash_changes_when_member_changes(saved_run):
    env = saved_run
    original = env.descriptor["files_sha256"]["fold_models/fold_0.ubj"]
    (env.root / "fold_models/fold_0.ubj").write_text("different")
    assert file_sha256(env.root / "fold_models/fold_0.ubj") != original


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b"])
def test_unsafe_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_file(tmp_path, name)


def test_default_evaluation_stays_final():
    assert EvaluationConfig(parent_run_id="run_one").predictor.kind == "final"


def test_cv_mean_changes_evaluation_identity():
    final = EvaluationConfig(parent_run_id="run_one")
    folds = EvaluationConfig(parent_run_id="run_one", predictor={"kind": "cv_mean"})
    assert final.digest() != folds.digest()


@pytest.mark.parametrize(
    "extras",
    [
        {"calibration_id": "cal_wrong"},
        {"release_variant": "calibrated", "calibration_id": "cal_wrong"},
        {"policy": {"kind": "threshold", "value": 0.5}},
    ],
)
def test_final_model_calibration_policy_not_reused_for_cv_mean(extras):
    with pytest.raises(ValidationError):
        EvaluationConfig(parent_run_id="run_one", predictor={"kind": "cv_mean"}, **extras)


def test_bound_fold_evaluation_loader_and_release_constructor_match(
    saved_run, tmp_path, monkeypatch
):
    pytest.importorskip("ml_template.training.run_config")
    from ml_template.models import model_factory
    from ml_template.models.predictor_selection import load_evaluation_predictor
    from ml_template.scoring.input_contract import frame_schema
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.release_artifacts import runtime_info, seal, verify, write_json
    from ml_template.training.run_config import RunConfig

    env = saved_run
    cfg = RunConfig.model_validate(
        {
            "experiment_name": "cv_mean_fixture",
            "data": {
                "dataset_name": "fixture",
                "dataset_config": "configs/unused.yaml",
                "silver_dataset_version_id": "silver_fixture",
                "feature_set_id": None,
            },
            "split": {"n_folds": 3},
            "fitted_features": [
                {"step": "power_transform", "columns": ["x"], "method": "yeo-johnson"}
            ],
        }
    )
    monkeypatch.setattr(model_factory, "load_adapter", env.loader)
    X = pd.DataFrame({"x": [0.1, np.nan, 0.9]}, index=pd.Index([30, 10, 20], name="entity"))
    prepared = SimpleNamespace(
        run_cfg=cfg,
        run_dir=env.root,
        spec=SimpleNamespace(predictor=SimpleNamespace(kind="cv_mean")),
        provenance={"predictor": env.descriptor},
    )
    evaluation_model = load_evaluation_predictor(prepared)
    expected = evaluation_model.predict_proba(X)
    package = tmp_path / "release_package"
    package.mkdir()
    for name in required_predictor_files(env.descriptor):
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(env.root / name, target)
    policy = {"kind": "none"}
    write_json(
        package / "release.json",
        {
            "schema_version": 1,
            "parent_run_id": "run_fixture",
            "evaluation_id": "eval_fixture",
            "calibration_id": None,
            "release_variant": "raw",
            "predictor_kind": "cv_mean",
            "policy": policy,
        },
    )
    write_json(package / "predictor.json", env.descriptor)
    write_json(package / "config.json", cfg.model_dump(mode="json"))
    schema = frame_schema(X, missingness={"default_max_rate": 1.0})
    write_json(
        package / "contract.json",
        {
            "schema_version": 2,
            "key_col": "entity",
            "key_kind": "int",
            "source_schema": schema,
            "model_schema": schema,
        },
    )
    write_json(
        package / "feature_recipe.json",
        {"schema_version": 1, "raw_columns": ["x"], "columns": ["x"], "blocks": []},
    )
    write_json(package / "runtime.json", runtime_info())
    write_json(
        package / "evidence.json",
        {
            "provenance": {"predictor": env.descriptor},
            "evaluation_spec": {"predictor": {"kind": "cv_mean", "weighting": "equal"}},
        },
    )
    seal(package)
    assert not (package / "model.ubj").exists()
    verify(package)  # Fold-only packages must be valid without a dummy final model.
    predictor = ReleasedPredictor(package)
    got = predictor.predict(X.reset_index(), chunk_size=1)
    np.testing.assert_allclose(got.p_raw, expected, rtol=0, atol=1e-15)
    assert got.entity.tolist() == [30, 10, 20]
    np.testing.assert_array_equal(got.p_raw, got.p_release)
    reordered = predictor.predict(X.iloc[::-1].reset_index(), chunk_size=2).set_index("entity")
    np.testing.assert_allclose(reordered.loc[X.index, "p_raw"], expected, rtol=0, atol=1e-15)

    # Re-sealing a changed predictor is insufficient if its evaluated identity differs.
    changed = copy.deepcopy(env.descriptor)
    member_name = changed["members"][0]["model_path"]
    (package / member_name).write_text("changed-after-evaluation")
    changed["files_sha256"][member_name] = file_sha256(package / member_name)
    write_json(package / "predictor.json", changed)
    seal(package)
    with pytest.raises(ValueError, match="evaluated member identity"):
        ReleasedPredictor(package)
