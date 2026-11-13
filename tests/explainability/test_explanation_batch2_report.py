import pandas as pd

from ml_template.explainability.report import write_report


def test_report_escapes_untrusted_feature_names(tmp_path):
    summary = {
        "explanation_id": "explain_fixture",
        "release_id": "rel_fixture",
        "scoring_batch_id": "score_fixture",
        "n_rows": 3,
        "n_features": 1,
    }
    ranking = pd.DataFrame(
        {"feature": ["<script>alert(1)</script>|x"], "mean_abs_contribution": [0.2]}
    )
    write_report(tmp_path, summary, ranking)
    html = (tmp_path / "report.html").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "no labels" in html
