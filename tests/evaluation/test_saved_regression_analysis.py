"""Saved-run analysis integration: optional PyArrow/SQLite dependencies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_saved_analysis_reads_development_only_and_preserves_parent(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    sa = pytest.importorskip("sqlalchemy")
    from ml_template import paths
    from ml_template.db import model_run_registry
    from ml_template.evaluation.regression_analysis_run import run_saved_analysis
    from ml_template.scoring.release_artifacts import sha256
    from ml_template.training.experiment_runner import run_experiment
    from ml_template.training.run_config import RunConfig
    from ml_template.training.split import make_split, save_split

    sr, sp = tmp_path / "silver", tmp_path / "splits"
    monkeypatch.setattr(paths, "DATA_SILVER", sr)
    monkeypatch.setattr(paths, "DATA_SPLITS", sp)
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "data: {task: regression, entity_id: id, target: strength}\n"
        "bronze_source_contract:\n  roles: {id: [id], feature: [x], target: [strength]}\n"
        "silver_eda_contract:\n  canonical_names: {id: id, x: x, strength: strength}\n"
    )
    cfg = RunConfig.model_validate(
        {
            "task": "regression",
            "experiment_name": "analysis_test",
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
    df = pd.DataFrame({"id": np.arange(100), "x": np.linspace(-2.0, 5.0, 100)})
    df["strength"] = 2 * df.x + 0.25
    sd = sr / "test/silver_test"
    sd.mkdir(parents=True)
    df.to_parquet(sd / "data.parquet", index=False)
    (sd / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": "silver_test",
                "validation_passed": True,
                "roles": {"id": ["id"], "feature": ["x"], "target": ["strength"]},
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "parquet_sha256": sha256(sd / "data.parquet"),
            }
        )
    )
    assignment = make_split(df, cfg.split, "id", "strength")
    save_split(assignment, "silver_test", cfg.split, "id", "strength", df.strength, sp)

    class Fake:
        max_rounds = 20

        def fit_with_validation(self, Xt, yt, Xv, yv):
            return 3

        def refit(self, X, y, n_rounds):
            self.mean = float(y.mean())

        def predict(self, X):
            return np.full(len(X), self.mean)

        def save(self, path):
            path.write_text("fake model")

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    trained = run_experiment(
        [cfg],
        engine,
        kind="single",
        run_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        silver_root=sr,
        split_root=sp,
        factory_builder=lambda cfg: Fake,
    )
    assert not trained["failures"]
    rid = trained["run_ids"][0]
    run_dir = tmp_path / "runs" / rid
    before = {str(p.relative_to(run_dir)): sha256(p) for p in run_dir.rglob("*") if p.is_file()}
    original = pd.read_parquet
    stage = {"execute": False}
    dev_keys = set(assignment.loc[assignment.partition == "dev", "id"])
    reads = []

    def spy(path, *args, **kwargs):
        if Path(path).name == "data.parquet":
            assert stage["execute"]
            filters = kwargs.get("filters")
            assert filters and set(filters[0][2]) == dev_keys
        result = original(path, *args, **kwargs)
        if Path(path).name == "data.parquet":
            assert set(result.id) == dev_keys
            reads.append(set(result.id))
        return result

    monkeypatch.setattr(pd, "read_parquet", spy)
    output_root = tmp_path / "analyses"
    checked = run_saved_analysis(
        engine, rid, output_root=output_root, permutation=False, plots=False
    )
    assert (
        checked["ready"] and not checked["development_labels_loaded"] and not output_root.exists()
    )
    stage["execute"] = True
    completed = run_saved_analysis(
        engine,
        rid,
        check_only=False,
        output_root=output_root,
        permutation=False,
        plots=False,
        target_unit="MPa",
    )
    assert completed["status"] == "succeeded" and reads
    directory = Path(completed["artifact_dir"])
    assert set(original(directory / "oof_residuals.parquet").id) == dev_keys
    after = {str(p.relative_to(run_dir)): sha256(p) for p in run_dir.rglob("*") if p.is_file()}
    assert before == after
    assert model_run_registry.get_run(engine, rid)["holdout_evaluated_at"] is None
