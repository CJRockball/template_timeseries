"""Synthetic keyed data and explicitly mocked Parquet transport."""

import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest
from scipy.special import expit

from ml_template.explainability.artifacts import fingerprint, seal_explanation, write_json
from ml_template.explainability.presentation_data import load_explanation


@pytest.fixture
def presentation_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "explanation"
    directory.mkdir()
    index = pd.Index([30, 10, 20], name="entity")
    features = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "kind": pd.Categorical(["L", "H", "L"])}, index=index
    )
    contributions = pd.DataFrame({"x": [0.4, -0.2, 0.3], "kind": [0.1, 0.3, -0.1]}, index=index)
    margin = -1 + contributions.sum(axis=1)
    rows = pd.DataFrame(
        {
            "base_value": -1.0,
            "margin": margin,
            "p_raw": expit(margin),
            "p_release": expit(margin) * 0.5,
            "p_calibrated": expit(margin) * 0.5,
            "policy_flag": [True, False, False],
            "explanation_id": "explain_fixture",
            "release_id": "rel_fixture",
            "scoring_batch_id": "score_fixture",
        },
        index=index,
    )
    importance = pd.DataFrame(
        {
            "feature": contributions.columns,
            "mean_abs_contribution": contributions.abs().mean().to_numpy(),
            "mean_contribution": contributions.mean().to_numpy(),
            "n_rows": 3,
        }
    )
    tables = {
        "rows.parquet": rows,
        "features.parquet": features,
        "contributions.parquet": contributions,
        "global_summary.parquet": importance,
    }
    for name in tables:
        (directory / name).write_bytes(b"fake Parquet transport: unit tests only")
    binding = {
        "release_id": "rel_fixture",
        "scoring_batch_id": "score_fixture",
        "source": "fixture",
    }
    binding["binding_sha256"] = fingerprint(binding)
    summary = {
        "schema_version": 1,
        "artifact_state": "complete",
        "explanation_id": "explain_fixture",
        "release_id": "rel_fixture",
        "scoring_batch_id": "score_fixture",
        "n_rows": 3,
        "n_features": 2,
        "labels_loaded": False,
        "persisted_scoring_batch_verified": True,
        "output_scale": "base_model_log_odds",
    }
    metadata = {"features": [{"name": name, "kind": "raw_input"} for name in features.columns]}
    values = {
        "summary.json": summary,
        "feature_metadata.json": metadata,
        "verification.json": {
            "passed": True,
            "tolerance": {
                "margin_atol": 1e-5,
                "margin_rtol": 1e-5,
                "probability_atol": 1e-6,
                "probability_rtol": 1e-5,
            },
        },
        "provenance.json": {"source_binding": binding},
        "config.json": {},
    }
    for name, value in values.items():
        write_json(directory / name, value)
    (directory / "report.md").write_text("original report")
    (directory / "report.html").write_text("original report")
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE explanation_runs (explanation_id TEXT PRIMARY KEY,release_id TEXT,scoring_batch_id TEXT,status TEXT,started_at TEXT,artifact_dir TEXT,n_rows INTEGER,n_features INTEGER,content_sha256 TEXT,binding_sha256 TEXT)"
        )
        conn.execute(
            "INSERT INTO explanation_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "explain_fixture",
                "rel_fixture",
                "score_fixture",
                "succeeded",
                "2026-10-20",
                str(directory),
                3,
                2,
                "pending",
                binding["binding_sha256"],
            ),
        )

    def reseal():
        for name, value in values.items():
            write_json(directory / name, value)
        manifest = seal_explanation(directory, "explain_fixture")
        with sqlite3.connect(database) as conn:
            conn.execute(
                "UPDATE explanation_runs SET content_sha256=?", (manifest["content_sha256"],)
            )
        return manifest

    reseal()
    original = pd.read_parquet

    def read(path, *args, **kwargs):
        if str(path).startswith(str(directory)) and path.name in tables:
            return tables[path.name].copy()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    return SimpleNamespace(
        directory=directory,
        database=database,
        tables=tables,
        values=values,
        summary=summary,
        reseal=reseal,
        view=lambda: load_explanation(database, "explain_fixture"),
    )
