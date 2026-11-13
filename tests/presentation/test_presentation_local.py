"""Opt-in real Parquet/artifact report check; no inference or source mutation."""

import os
from pathlib import Path

import pytest


@pytest.mark.local_data
def test_local_report_export(tmp_path):
    identity = os.environ.get("ML_TEMPLATE_PRESENTATION_EXPLANATION_ID")
    if not identity:
        pytest.skip("set ML_TEMPLATE_PRESENTATION_EXPLANATION_ID for real-artifact validation")
    from ml_template.explainability.artifacts import file_entries
    from ml_template.explainability.export_report import export_report
    from ml_template.explainability.presentation_data import load_explanation
    from ml_template.paths import ROOT

    database = Path(os.environ.get("ML_TEMPLATE_PRESENTATION_DATABASE", ROOT / "data/project.db"))
    view = load_explanation(database, identity)
    before = file_entries(view.record["artifact_dir"])
    result = export_report(database, identity, tmp_path / "reports")
    assert (Path(result["artifact_dir"]) / "report.html").stat().st_size > 0
    assert before == file_entries(view.record["artifact_dir"])
