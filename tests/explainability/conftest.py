"""Batch-two fixtures: fake model/Parquet bytes, real content manifests and SQLite."""

import sqlite3
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_template.explainability.artifacts import file_entries, sha256, write_json
from ml_template.explainability.explanation_config import ExplanationConfig
from ml_template.scoring.release_artifacts import seal


@pytest.fixture
def batch2_layout(tmp_path, monkeypatch):
    release, score = tmp_path / "release", tmp_path / "score"
    release.mkdir()
    score.mkdir()
    pinned = tmp_path / "pinned/ml_template"
    pinned.mkdir(parents=True)
    (pinned / "__init__.py").write_text("")
    info = {
        "schema_version": 1,
        "parent_run_id": "run_fixture",
        "calibration_id": None,
        "release_variant": "raw",
        "policy": {"kind": "none", "value": None, "batch_definition": "entire batch"},
    }
    contract = {
        "schema_version": 1,
        "key_col": "entity",
        "key_kind": "int",
        "source_schema": [{"name": "x", "kind": "float", "dtype": "float64"}],
        "model_schema": [{"name": "x", "kind": "float", "dtype": "float64"}],
    }
    recipe = {"schema_version": 1, "raw_columns": ["x"], "columns": ["x"], "blocks": []}
    for name, value in {
        "release.json": info,
        "contract.json": contract,
        "feature_recipe.json": recipe,
        "config.json": {},
        "runtime.json": {},
        "evidence.json": {},
    }.items():
        write_json(release / name, value)
    (release / "model.ubj").write_bytes(b"NOT A REAL MODEL")
    manifest = seal(release)
    frame = pd.DataFrame({"entity": [30, 10, 20], "x": [1.0, 2.0, 3.0]})
    frame.to_csv(score / "input.csv", index=False)
    predictions = pd.DataFrame(
        {
            "entity": [20, 30, 10],
            "p_raw": [0.3, 0.1, 0.2],
            "p_release": [0.3, 0.1, 0.2],
            "release_id": manifest["release_id"],
            "scoring_batch_id": "score_fixture",
        }
    )
    (score / "predictions.parquet").write_bytes(b"FAKE PARQUET: mocked read only")
    summary = {
        "schema_version": 1,
        "scoring_batch_id": "score_fixture",
        "release_id": manifest["release_id"],
        "content_sha256": manifest["content_sha256"],
        "parent_run_id": info["parent_run_id"],
        "calibration_id": None,
        "release_variant": "raw",
        "policy": info["policy"],
        "batch_definition": "entire batch",
        "batch_definition_acknowledged": False,
        "labels_loaded": False,
        "n_rows": 3,
        "input_sha256": sha256(score / "input.csv"),
        "n_flagged": None,
        "flagged_fraction": None,
    }
    write_json(score / "summary.json", summary)

    def reseal():
        write_json(score / "summary.json", summary)
        write_json(score / "artifacts.json", {"files": file_entries(score)})

    reseal()
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE scoring_batches (scoring_batch_id TEXT PRIMARY KEY, release_id TEXT,status TEXT,input_sha256 TEXT,n_rows INTEGER,n_flagged INTEGER)"
        )
        conn.execute(
            "INSERT INTO scoring_batches VALUES (?,?,?, ?,?,?)",
            (
                "score_fixture",
                manifest["release_id"],
                "succeeded",
                summary["input_sha256"],
                3,
                None,
            ),
        )
    config = ExplanationConfig(
        release_dir=str(release),
        scoring_dir=str(score),
        expected_release_id=manifest["release_id"],
        expected_scoring_batch_id="score_fixture",
        runtime_python=sys.executable,
        runtime_source_root=str(pinned),
        registry_path=str(database),
        output_root=str(tmp_path / "outputs"),
    ).resolved(tmp_path)
    original_read = pd.read_parquet

    def read_parquet(path, *args, **kwargs):
        if str(path) == str(score / "predictions.parquet"):
            return predictions.copy()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    return SimpleNamespace(
        config=config,
        frame=frame,
        predictions=predictions,
        summary=summary,
        contract=contract,
        info=info,
        release=release,
        score=score,
        database=database,
        reseal=reseal,
        manifest=manifest,
        pinned=pinned,
    )
