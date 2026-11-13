"""Opt-in actual model preflight; never executes real holdout evaluation."""

import os

import pytest


@pytest.mark.local_data
def test_local_preflight():
    if os.environ.get("FINAL_EVALUATION_LOCAL_CHECK") != "1":
        pytest.skip("set FINAL_EVALUATION_LOCAL_CHECK=1 to validate local provenance")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pyarrow")
    from ml_template.db.connection import get_engine
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.evaluation_config import load_evaluation_config

    result = evaluate_spec(
        get_engine(), load_evaluation_config("configs/evaluation/test_958bec.yaml"), check_only=True
    )
    assert result["holdout_labels_loaded"] is False
    assert result.get("n_holdout_rows", result.get("summary", {}).get("n_rows")) == 2000
