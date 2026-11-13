"""Regression release, parity, runtime, batch scoring and keyed export."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_template.scoring.export_predictions import export_frame
from ml_template.scoring.regression_release_config import RegressionReleaseConfig


def test_continuous_export_preserves_reference_order():
    prediction = pd.DataFrame({"id": [2, 1], "y_pred_release": [-3.5, 40.25]})
    reference = pd.DataFrame({"id": [1, 2], "strength": [0.0, 0.0]})
    out = export_frame(
        prediction,
        reference,
        key="id",
        score="y_pred_release",
        output_column="strength",
        task="regression",
    )
    assert out.id.tolist() == [1, 2]
    assert out.strength.tolist() == [40.25, -3.5]
    with pytest.raises(ValueError):
        export_frame(
            prediction, reference, key="id", score="y_pred_release", output_column="strength"
        )


@pytest.mark.parametrize("bad", [[np.inf, 1.0], [np.nan, 1.0], [True, False]])
def test_export_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        export_frame(
            pd.DataFrame({"id": [1, 2], "y": bad}),
            pd.DataFrame({"id": [1, 2]}),
            key="id",
            score="y",
            output_column="strength",
            task="regression",
        )


def test_release_requires_explicit_research_acknowledgement():
    with pytest.raises(ValueError):
        RegressionReleaseConfig(
            name="concrete", evaluation_id="eval_1", parent_run_id="run_1", intended_use="Research"
        )
    cfg = RegressionReleaseConfig(
        name="concrete",
        evaluation_id="eval_1",
        parent_run_id="run_1",
        intended_use="Research",
        acknowledge_research_release=True,
    )
    assert cfg.task == "regression" and cfg.schema_version == 2


@pytest.fixture
def evaluated_world(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    sa = pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template import paths
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.scoring.release_artifacts import sha256
    from ml_template.training.experiment_runner import run_experiment
    from ml_template.training.run_config import RunConfig
    from ml_template.training.split import make_split, save_split

    sr, sp = tmp_path / "silver", tmp_path / "splits"
    monkeypatch.setattr(paths, "DATA_SILVER", sr)
    monkeypatch.setattr(paths, "DATA_SPLITS", sp)
    df = pd.DataFrame({"id": np.arange(120), "x": np.linspace(-5.0, 20.0, 120)})
    df["strength"] = 2 * df.x + 0.25
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: strength}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [x], target: [strength]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, x: x, strength: strength}\n"
        "  canonical_dtypes: {id: int64, x: float64, strength: float64}\n"
        "  value_mappings: {}\n  allowed_values: {}\n  missingness: {default_max_rate: 0.0}\n"
    )
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "release_test",
            "data": {
                "dataset_name": "test",
                "dataset_config": str(contract),
                "silver_dataset_version_id": "silver_test",
            },
            "split": {"method": "random", "n_folds": 3},
            "model": {
                "type": "xgb",
                "params": {
                    "objective": "reg:squarederror",
                    "n_estimators": 12,
                    "early_stopping_rounds": 3,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                },
            },
        }
    )
    sd = sr / "test/silver_test"
    sd.mkdir(parents=True)
    df.to_parquet(sd / "data.parquet", index=False)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "validation_passed": True,
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "config_sha256": sha256(contract),
                "roles": {"id": ["id"], "feature": ["x"], "target": ["strength"]},
                "parquet_sha256": sha256(sd / "data.parquet"),
            }
        )
    )
    assign = make_split(df, cfg.split, "id", "strength")
    save_split(assign, "silver_test", cfg.split, "id", "strength", df.strength, sp)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    trained = run_experiment(
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
    evaluated = evaluate_spec(
        engine, RegressionEvaluationConfig(parent_run_id=rid), output_root=tmp_path / "evaluations"
    )
    request = RegressionReleaseConfig(
        name="concrete",
        parent_run_id=rid,
        evaluation_id=evaluated["evaluation_id"],
        intended_use="Synthetic integration test",
        acknowledge_research_release=True,
    )
    return engine, request, df[["id", "x"]], tmp_path


def test_release_parity_never_loads_labels(evaluated_world, monkeypatch):
    from ml_template.scoring.release_builder import build_release

    engine, request, frame, root = evaluated_world
    original = pd.read_parquet
    reads = []

    def spy(path, *args, **kwargs):
        columns = kwargs.get("columns")
        if Path(path).name in {"data.parquet", "holdout_predictions.parquet"}:
            assert columns is not None and not set(columns) & {"strength", "y_true"}
            reads.append(tuple(columns))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    checked = build_release(engine, request, check_only=True, output_root=root / "releases")
    assert checked["ready"] and not checked["parity_executed"]
    result = build_release(engine, request, output_root=root / "releases")
    assert result["parity"]["metrics_recomputed"] is False
    assert result["parity"]["labels_loaded"] is False
    assert ("id", "y_pred") in reads


def test_release_scoring_roundtrip_and_runtime_guards(evaluated_world, monkeypatch):
    from ml_template.scoring import release_artifacts
    from ml_template.scoring.batch import score_batch
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.release_builder import build_release

    engine, request, frame, root = evaluated_world
    built = build_release(engine, request, output_root=root / "releases")
    release = Path(built["artifact_dir"])
    predictor = ReleasedPredictor(release, expected_release_id=built["release_id"])
    whole = predictor.predict(frame)
    chunked = predictor.predict(frame.iloc[::-1], chunk_size=7)
    np.testing.assert_allclose(
        whole.set_index("id").sort_index().y_pred_release,
        chunked.set_index("id").sort_index().y_pred_release,
        rtol=0,
        atol=1e-7,
    )
    assert "p_release" not in whole and "policy_flag" not in whole
    with pytest.raises(ValueError):
        predictor.predict(frame.assign(strength=0.0))
    with pytest.raises(ValueError):
        predictor.predict(pd.concat([frame.iloc[:1], frame.iloc[:1]]))
    canonical = root / "unlabelled.parquet"
    frame.to_parquet(canonical, index=False)
    checked = score_batch(None, release, canonical, check_only=True)
    assert checked["ready"] and not checked["predictions_computed"]
    scored = score_batch(engine, release, canonical, output_root=root / "scores", chunk_size=7)
    output = pd.read_parquet(Path(scored["artifact_dir"]) / "predictions.parquet")
    assert output.id.tolist() == frame.id.tolist() and "y_pred_release" in output
    summary = json.loads((Path(scored["artifact_dir"]) / "summary.json").read_text())
    assert summary["task"] == "regression" and summary["labels_loaded"] is False
    # Sealed inference must not consult the training registry.
    from ml_template.db import model_run_registry

    monkeypatch.setattr(
        model_run_registry,
        "get_run",
        lambda *a: (_ for _ in ()).throw(AssertionError("Inference queried registry")),
    )
    ReleasedPredictor(release).predict(frame.iloc[:2])
    real_runtime = release_artifacts.runtime_info
    monkeypatch.setattr(
        release_artifacts, "runtime_info", lambda: {**real_runtime(), "python": "unexpected"}
    )
    with pytest.raises(ValueError, match="runtime"):
        ReleasedPredictor(release)
    monkeypatch.setattr(release_artifacts, "runtime_info", real_runtime)
    descriptor = json.loads((release / "predictor.json").read_text())
    member = release / descriptor["members"][0]["model_path"]
    member.write_text("tampered")
    with pytest.raises(ValueError):
        ReleasedPredictor(release)


def test_parity_projects_only_keys_and_predictions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ml_template.scoring.regression_release import check_regression_parity
    from ml_template.scoring.release_artifacts import sha256

    path = tmp_path / "holdout_predictions.parquet"
    path.write_bytes(b"projection test placeholder")
    source = pd.DataFrame(
        {"x": [-2.0, 0.0, 1.0, 4.0, 8.0, 9.0]}, index=pd.Index([1, 2, 3, 4, 5, 6], name="id")
    )
    reference = pd.DataFrame({"id": source.index, "y_pred": 2 * source.x.to_numpy()})

    def read(path, *, columns):
        assert columns == ["id", "y_pred"]
        return reference[columns].copy()

    monkeypatch.setattr(pd, "read_parquet", read)
    chunks = []

    class Predictor:
        def predict(self, frame, *, chunk_size=None):
            chunks.append(chunk_size)
            return pd.DataFrame(
                {
                    "id": frame.id.to_numpy(),
                    "y_pred_raw": 2 * frame.x.to_numpy(),
                    "y_pred_release": 2 * frame.x.to_numpy(),
                }
            )

    material = {
        "prepared": SimpleNamespace(key_col="id", hold_keys=list(source.index)),
        "evaluation_dir": tmp_path,
        "source": source,
        "reference_sha256": sha256(path),
    }
    result = check_regression_parity(Predictor(), material, 1e-7)
    assert result["max_abs_diff"] == 0 and result["labels_loaded"] is False
    assert result["metrics_recomputed"] is False and any(
        c is not None and c < len(source) for c in chunks
    )
