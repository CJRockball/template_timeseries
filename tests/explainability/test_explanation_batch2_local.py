"""Opt-in real release + real pinned interpreter + real Parquet smoke test.

This creates an explanation execution and artifacts; it does not evaluate labels.
"""

import os
from pathlib import Path

import pytest


@pytest.mark.local_data
def test_real_saved_batch_explanation():
    config_path = os.environ.get("ML_TEMPLATE_EXPLANATION_CONFIG")
    if not config_path:
        pytest.skip("set ML_TEMPLATE_EXPLANATION_CONFIG for an intentional real-artifact smoke run")
    from ml_template.explainability.artifacts import file_entries, verify_explanation
    from ml_template.explainability.explain_batch import explain_batch
    from ml_template.explainability.explanation_config import load_config
    from ml_template.paths import ROOT

    config = load_config(config_path, project_root=ROOT)
    before = file_entries(config.release_dir), file_entries(config.scoring_dir)
    assert explain_batch(config, check_only=True)["ready"]
    result = explain_batch(config)
    assert result["status"] == "succeeded"
    manifest = verify_explanation(Path(result["artifact_dir"]), result["explanation_id"])
    assert manifest["files"]["rows.parquet"]["bytes"] > 0
    assert before == (file_entries(config.release_dir), file_entries(config.scoring_dir))
