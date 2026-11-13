import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import calibration_data as cd

CID = "cal_20261017_063551_260e14"
PARENT = "run_20261016_121745_958bec"


def write_manifest(folder):
    files = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in folder.iterdir()
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    (folder / "artifact_manifest.json").write_text(json.dumps({"files": files}))


@pytest.fixture
def fixture_bundle(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE calibration_runs (calibration_id TEXT, parent_run_id TEXT, status TEXT, started_at TEXT, finished_at TEXT, method TEXT, config_json TEXT, artifact_dir TEXT, error_summary TEXT)"
    )
    relative = "artifacts/calibration/" + CID
    conn.execute(
        "INSERT INTO calibration_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            CID,
            PARENT,
            "succeeded",
            "2026-10-17T06:35:51+00:00",
            None,
            "sigmoid",
            "{}",
            relative,
            None,
        ),
    )
    conn.commit()
    folder = tmp_path / relative
    folder.mkdir(parents=True)
    for name, value in {
        "config.json": {"parent_run_id": PARENT, "method": "sigmoid"},
        "provenance.json": {
            "parent_run_id": PARENT,
            "key_col": "udi",
            "feature_set_id": "fs_d0383e096267",
            "parent_holdout_evaluated_at": None,
        },
        "calibrator.json": {"method": "sigmoid", "epsilon": 1e-7, "slope": 0.8, "intercept": -1},
        "diagnostic_notes.json": {"warning_slope_at_floor": False},
    }.items():
        (folder / name).write_text(json.dumps(value))
    metrics = pd.DataFrame(
        [
            {"split": cd.SPLIT, "variant": variant, "fold": fold, "metric": metric, "value": value}
            for variant, value in [("raw", 0.2), ("calibrated_crossfit", 0.1)]
            for fold in [-1, 0, 1]
            for metric in ["brier", "log_loss"]
        ]
    )
    metrics.to_csv(folder / "metrics.csv", index=False)
    reliability = pd.DataFrame(
        [
            {
                "variant": variant,
                "bin": 0,
                "n_rows": 6,
                "n_positives": 2,
                "mean_probability": p,
                "event_rate": 2 / 6,
                "event_rate_low": 0.1,
                "event_rate_high": 0.8,
            }
            for variant, p in [("raw", 0.5), ("calibrated_crossfit", 0.35)]
        ]
    )
    reliability.to_csv(folder / "reliability.csv", index=False)
    write_manifest(folder)
    yield conn, tmp_path, folder
    conn.close()


def test_missing_registry_does_not_create_tables():
    conn = sqlite3.connect(":memory:")
    assert cd.list_calibrations(conn).empty
    assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    conn.close()


def test_list_and_parent_filter(fixture_bundle):
    conn, _, _ = fixture_bundle
    assert cd.list_calibrations(conn).calibration_id.tolist() == [CID]
    assert len(cd.list_calibrations(conn, PARENT)) == 1
    assert cd.list_calibrations(conn, "run_wrong").empty


def test_read_only_and_verified_artifacts(fixture_bundle):
    conn, root, folder = fixture_bundle
    conn.execute("PRAGMA query_only = ON")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    bundle = cd.load_calibration(conn, CID, repo_root=root, include_predictions=False)
    assert len(bundle["metrics"]) == 12
    assert len(bundle["reliability"]) == 2
    assert bundle["provenance"]["feature_set_id"] == "fs_d0383e096267"
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_metric_delta_sign(fixture_bundle):
    conn, root, _ = fixture_bundle
    bundle = cd.load_calibration(conn, CID, repo_root=root, include_predictions=False)
    table = cd.metric_comparison(bundle["metrics"])
    assert np.allclose(table.delta_cal_minus_raw, -0.1)
    assert table.direction.eq("lower is better").all()


def test_tampered_artifact_is_suppressed(fixture_bundle):
    conn, root, folder = fixture_bundle
    (folder / "metrics.csv").write_text("tampered")
    bundle = cd.load_calibration(conn, CID, repo_root=root, include_predictions=False)
    assert bundle["metrics"].empty
    assert not bundle["reliability"].empty
    assert any("hash mismatch" in w for w in bundle["warnings"])


def test_missing_manifest_suppresses_all_diagnostics(fixture_bundle):
    conn, root, folder = fixture_bundle
    (folder / "artifact_manifest.json").unlink()
    bundle = cd.load_calibration(conn, CID, repo_root=root)
    assert bundle["metrics"].empty and bundle["calibrator"] == {}
    assert bundle["warnings"]


def test_wrong_parent_suppresses_diagnostics(fixture_bundle):
    conn, root, folder = fixture_bundle
    (folder / "provenance.json").write_text(json.dumps({"parent_run_id": "run_other"}))
    write_manifest(folder)
    bundle = cd.load_calibration(conn, CID, repo_root=root)
    assert bundle["metrics"].empty and bundle["calibrator"] == {}
    assert any("parent binding" in w for w in bundle["warnings"])


def test_failed_run_is_visible_but_not_loaded(fixture_bundle):
    conn, root, _ = fixture_bundle
    conn.execute("UPDATE calibration_runs SET status='failed', error_summary='example failure'")
    bundle = cd.load_calibration(conn, CID, repo_root=root)
    assert bundle["metrics"].empty
    assert "example failure" in bundle["warnings"]


def test_slope_warning(fixture_bundle):
    conn, root, folder = fixture_bundle
    (folder / "diagnostic_notes.json").write_text(json.dumps({"warning_slope_at_floor": True}))
    write_manifest(folder)
    bundle = cd.load_calibration(conn, CID, repo_root=root, include_predictions=False)
    assert any("lower bound" in w for w in bundle["warnings"])


def test_relocated_directory(fixture_bundle):
    conn, root, _ = fixture_bundle
    conn.execute("UPDATE calibration_runs SET artifact_dir='/missing/old/checkout'")
    bundle = cd.load_calibration(conn, CID, repo_root=root, include_predictions=False)
    assert not bundle["metrics"].empty
    assert any("relocated" in w for w in bundle["warnings"])


def test_prediction_validation_and_projection():
    frame = pd.DataFrame(
        {
            "udi": [1, 2, 3],
            "fold": [0, 1, 0],
            "y_true": [0, 0, 1],
            "p_raw": [0, 0.5, 1],
            "p_cal_crossfit": [0.1, 0.2, 0.8],
            "p_cal_fit": [0.1, 0.2, 0.9],
        }
    )
    clean = cd.validate_predictions(frame, "udi")
    assert "p_cal_fit" not in clean
    hist = cd.distribution_table(clean, n_bins=4)
    for variant in cd.VARIANTS:
        assert hist.loc[hist.variant.eq(variant), "count"].sum() == 3
    frame.loc[0, "fold"] = -1
    with pytest.raises(ValueError, match="development folds"):
        cd.validate_predictions(frame, "udi")


def test_mocked_parquet_projection(fixture_bundle, monkeypatch):
    conn, root, folder = fixture_bundle
    (folder / "development_predictions.parquet").write_bytes(
        b"synthetic placeholder; reader mocked"
    )
    write_manifest(folder)
    calls = []

    def read_parquet(path, *, columns):
        calls.append((path.name, columns))
        return pd.DataFrame(
            {
                "udi": [1, 2],
                "fold": [0, 1],
                "y_true": [0, 1],
                "p_raw": [0.2, 0.8],
                "p_cal_crossfit": [0.1, 0.7],
            }
        )

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    bundle = cd.load_calibration(conn, CID, repo_root=root)
    assert len(bundle["predictions"]) == 2
    assert calls == [
        ("development_predictions.parquet", ["udi", "fold", "y_true", "p_raw", "p_cal_crossfit"])
    ]


def test_sqlalchemy_engine_path(fixture_bundle):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    conn, _, _ = fixture_bundle
    engine = sqlalchemy.create_engine("sqlite://", creator=lambda: conn)
    assert len(cd.list_calibrations(engine)) == 1


def test_real_parquet_read(fixture_bundle):
    pytest.importorskip("pyarrow")
    conn, root, folder = fixture_bundle
    frame = pd.DataFrame(
        {
            "udi": [1, 2],
            "fold": [0, 1],
            "y_true": [0, 1],
            "p_raw": [0.2, 0.8],
            "p_cal_crossfit": [0.1, 0.7],
            "p_cal_fit": [0.1, 0.7],
        }
    )
    frame.to_parquet(folder / "development_predictions.parquet", index=False)
    write_manifest(folder)
    bundle = cd.load_calibration(conn, CID, repo_root=root)
    assert len(bundle["predictions"]) == 2
    assert "p_cal_fit" not in bundle["predictions"]
