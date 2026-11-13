import sqlite3

import pytest

from ml_template.db import scoring_registry as registry


def test_release_registration_and_batch_state():
    conn = sqlite3.connect(":memory:")
    info = {
        "name": "test",
        "parent_run_id": "run",
        "calibration_id": None,
        "evaluation_id": "eval",
        "created_at": "now",
        "policy": {"kind": "none"},
    }
    manifest = {"release_id": "rel", "content_sha256": "hash"}
    registry.register_release(conn, manifest, info, "/test")
    registry.register_release(conn, manifest, info, "/moved")
    assert conn.execute("SELECT count(*) FROM model_releases").fetchone()[0] == 1
    registry.start_batch(conn, "batch", "rel", "/batch", "input.csv")
    registry.finish_batch(conn, "batch", duration_s=1, n_rows=3, n_flagged=1)
    assert conn.execute("SELECT status,n_rows,n_flagged FROM scoring_batches").fetchone() == (
        "succeeded",
        3,
        1,
    )
    with pytest.raises(ValueError):
        registry.finish_batch(conn, "batch", duration_s=2)
    with pytest.raises(ValueError, match="conflict"):
        registry.register_release(
            conn, {"release_id": "rel", "content_sha256": "other"}, info, "/test"
        )


def test_failed_batch_retained():
    conn = sqlite3.connect(":memory:")
    registry.start_batch(conn, "batch", "rel", "/batch", "input.csv")
    registry.finish_batch(conn, "batch", duration_s=0.1, error="invalid input")
    assert conn.execute("SELECT status,error_summary FROM scoring_batches").fetchone() == (
        "failed",
        "invalid input",
    )
