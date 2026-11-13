"""Real synthetic regression FE train -> diagnostics -> evaluate -> release -> score."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
def test_native_regression_stateless_end_to_end(tmp_path, monkeypatch, engineered, predictor_kind):
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template import paths
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_analysis_run import run_saved_analysis
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.features.feature_set_store import build_block, create_feature_set
    from ml_template.scoring.batch import score_batch
    from ml_template.scoring.export_predictions import export_frame
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig
    from ml_template.scoring.release_artifacts import sha256
    from ml_template.scoring.release_builder import build_release
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
        "  canonical_dtypes: {id: int64, a: float64, b: int64, k: string, y: float64}\n"
        "  value_mappings: {}\n  allowed_values: {}\n  missingness: {default_max_rate: 0.0}\n"
    )
    df = pd.DataFrame(
        {
            "id": np.arange(120),
            "a": np.linspace(1.0, 6.0, 120),
            "b": np.tile([2, 3, 4], 40),
            "k": np.tile(["u", "v"], 60),
        }
    )
    df["y"] = 2 * df.a / df.b + (df.k == "u").astype(float) * 0.1
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
    if engineered:
        raw = df.set_index("id")[["a", "b", "k"]]
        block = build_block(
            "generic_interactions",
            raw,
            "silver_test",
            params={
                "numeric_pairs": [{"a": "a", "b": "b", "ops": ["ratio", "product"]}],
                "category_pairs": [{"a": "k", "b": "k"}],
            },
            root=fr,
        )
        fs = create_feature_set(
            "silver_test",
            [block["block_id"]],
            raw_columns=list(raw),
            columns=["a__div__b", "k__x__k", "a", "b"],
            root=fr,
        )
        fsid = fs["feature_set_id"]
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "stateless_test",
            "data": {
                "dataset_name": "test",
                "dataset_config": str(contract),
                "silver_dataset_version_id": "silver_test",
                "feature_set_id": fsid,
            },
            "split": {"method": "random", "n_folds": 3},
            "analysis": {
                "permutation_importance": False,
                "error_sample_size": 0,
                "learning_curves": False,
            },
            "model": {
                "type": "xgb",
                "params": {
                    "objective": "reg:squarederror",
                    "n_estimators": 20,
                    "early_stopping_rounds": 3,
                    "max_depth": 2,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                },
            },
        }
    )
    assignment = make_split(df, cfg.split, "id", "y")
    save_split(assignment, "silver_test", cfg.split, "id", "y", df.y, sp)
    # The existing runner's captured default feature_root is overridden in this temporary test only.
    original_loader = runner.load_run_inputs
    monkeypatch.setattr(
        runner,
        "load_run_inputs",
        lambda cfg, **kwargs: original_loader(cfg, feature_root=fr, **kwargs),
    )
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    trained = runner.run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=sr,
        split_root=sp,
    )
    assert not trained["failures"]
    rid = trained["run_ids"][0]
    source = sd / "data.parquet"
    original_read = pd.read_parquet
    stage = {"mode": "check"}
    dev = set(assignment.loc[assignment.partition == "dev", "id"])
    hold = set(assignment.loc[assignment.partition == "holdout", "id"])

    def spy(path, *args, **kwargs):
        if Path(path) == source and "y" in (kwargs.get("columns") or []):
            assert stage["mode"] in {"diagnostics", "evaluate"}
            keys = set(kwargs["filters"][0][2])
            assert keys == (dev if stage["mode"] == "diagnostics" else hold)
            assert not any("__" in name for name in kwargs["columns"])
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    spec = RegressionEvaluationConfig(parent_run_id=rid, predictor={"kind": predictor_kind})
    checked = evaluate_spec(engine, spec, check_only=True)
    assert checked["ready"] and checked["holdout_labels_loaded"] is False
    stage["mode"] = "diagnostics"
    diagnostic = run_saved_analysis(
        engine,
        rid,
        check_only=False,
        output_root=tmp_path / "analyses",
        repeats=2,
        max_rows=20,
        plots=False,
    )
    assert diagnostic["status"] == "succeeded"
    stage["mode"] = "evaluate"
    evaluated = evaluate_spec(engine, spec, output_root=tmp_path / "evaluations")
    request = RegressionReleaseConfig(
        name="synthetic_stateless",
        parent_run_id=rid,
        evaluation_id=evaluated["evaluation_id"],
        intended_use="Synthetic test",
        acknowledge_research_release=True,
    )
    stage["mode"] = "package"
    released = build_release(engine, request, output_root=tmp_path / "releases")
    assert (
        released["parity"]["labels_loaded"] is False and released["parity"]["max_abs_diff"] <= 1e-7
    )
    new = pd.DataFrame(
        {
            "id": [1001, 1002, 1003, 1004],
            "a": [1.5, 2.5, 3.5, 4.5],
            "b": [2, 3, 4, 2],
            "k": ["u", "v", "u", "v"],
        }
    )
    canonical = tmp_path / "unlabelled.parquet"
    new.to_parquet(canonical, index=False)
    from ml_template.scoring.pipeline import run_pipeline

    pipe = run_pipeline(
        engine,
        run_id=rid,
        input_path=canonical,
        evaluation_spec=spec,
        expected_evaluation_id=evaluated["evaluation_id"],
        release_dir=released["artifact_dir"],
        input_format="canonical",
        output_root=tmp_path / "pipelines",
        check_only=False,
        acknowledge_research_release=True,
        chunk_size=2,
    )
    assert pipe["status"] == "succeeded" and pipe["evaluation_reused"] is True
    # Released inference must not consult live feature sets, cached rows or raw datasets.
    shutil.move(str(fr), str(tmp_path / "features_offline"))
    predictor = ReleasedPredictor(released["artifact_dir"])
    whole = predictor.predict(new)
    shuffled = predictor.predict(new.iloc[::-1], chunk_size=1).set_index("id").sort_index()
    np.testing.assert_allclose(
        whole.set_index("id").sort_index().y_pred_release,
        shuffled.y_pred_release,
        rtol=0,
        atol=1e-7,
    )
    with pytest.raises(ValueError):
        predictor.predict(new.assign(y=0.0))
    with pytest.raises(ValueError):
        predictor.predict(new.assign(k="unknown"))
    canonical = tmp_path / "unlabelled.parquet"
    new.to_parquet(canonical, index=False)
    checked = score_batch(None, released["artifact_dir"], canonical, check_only=True)
    assert (
        checked["ready"]
        and checked["predictions_computed"] is False
        and checked["stateless_recipe_checked"]
    )
    scored = score_batch(
        engine, released["artifact_dir"], canonical, output_root=tmp_path / "scores", chunk_size=2
    )
    output = original_read(Path(scored["artifact_dir"]) / "predictions.parquet")
    exported = export_frame(
        output,
        pd.DataFrame({"id": new.id.iloc[::-1]}),
        key="id",
        score="y_pred_release",
        output_column="prediction",
        task="regression",
    )
    assert exported.id.tolist() == new.id.iloc[::-1].tolist()
    if engineered:
        bad = new.copy()
        bad.loc[0, "b"] = 0
        bad_path = tmp_path / "bad.parquet"
        bad.to_parquet(bad_path, index=False)
        with pytest.raises(ValueError):
            score_batch(None, released["artifact_dir"], bad_path, check_only=True)
