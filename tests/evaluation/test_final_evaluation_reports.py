import pytest

from ml_template.evaluation.evaluation_artifacts import manifest, verify, write_json
from ml_template.evaluation.final_report import regenerate_report, render_report


def minimal():
    return {
        "summary": {"evaluation_id": "e"},
        "spec": {},
        "warnings": ["<script>unsafe</script>"],
        "metrics": [],
        "differences": [],
        "policy_results": [],
        "reliability": [],
        "subgroups": [],
        "errors": [],
        "curves": [],
        "acceptance": {"criteria": []},
        "development_context": [],
        "provenance": {},
    }


def test_report_self_contained_and_escaped(tmp_path):
    render_report(minimal(), tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in page
    assert "<script>unsafe</script>" not in page
    assert "<script src=" not in page


def test_regeneration_no_database_and_no_source_changes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    write_json(source / "report_data.json", minimal())
    manifest(source)
    before = (source / "artifact_manifest.json").read_bytes()
    regenerate_report(source, target)
    assert (target / "report.html").exists()
    assert (source / "artifact_manifest.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        regenerate_report(source, target)


def test_integrity_failure(tmp_path):
    write_json(tmp_path / "report_data.json", minimal())
    manifest(tmp_path)
    (tmp_path / "report_data.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        verify(tmp_path)


def test_parquet_roundtrip_when_available(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    from ml_template.evaluation.evaluate_run import _save_predictions

    frame = pd.DataFrame({"id": [1], "y_true": [1], "p_raw": [0.5]})
    path = tmp_path / "holdout_predictions.parquet"
    _save_predictions(frame, path)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(path))


def test_existing_reliability_helper_when_available():
    diagnostics = pytest.importorskip("ml_template.evaluation.calibration_diagnostics")
    from ml_template.evaluation.evaluation_diagnostics import reliability_rows

    rows = reliability_rows([0, 0, 1, 1], {"raw": [0.2] * 4}, 5)
    assert len(rows) == 1 and rows[0]["n_rows"] == 4
    assert diagnostics.reliability_table([0, 0, 1, 1], [0.2] * 4, 5).shape[0] == 1
