import json
import math
import sqlite3
from pathlib import Path

import pytest
from forecasting_dashboard08_fixtures import fixture

from ml_template.dashboard import forecasting_data as data
from ml_template.dashboard.forecasting_figs import figures


def state(paths):
    return data.snapshot(paths["database"], paths["identifier"])


def test_summary_inventory(tmp_path):
    paths = fixture(tmp_path)
    assert data.inventory(paths["database"])[0][0]["value"] == paths["identifier"]
    result = data.view(state(paths))
    assert len(result["table"]) == 3
    assert next(r for r in result["table"] if r["method"] == "xgb")["rmse"] == pytest.approx(
        math.sqrt(5)
    )


def test_filtered_metrics_are_request_weighted(tmp_path):
    paths = fixture(tmp_path)
    snapshot = state(paths)
    rows = data.detail(paths["database"], snapshot)
    model = data.view(snapshot, rows=rows, methods=["xgb"])["table"][0]
    assert model["rmse"] == pytest.approx(math.sqrt(5)) and model["rmse"] != 2
    assert model["bias"] == 2
    filtered = data.view(
        snapshot, rows=rows, methods=["xgb"], series=['["b"]'], origins=["2020-02-17"]
    )["table"][0]
    assert filtered["n"] == 1 and filtered["rmse"] == 3


def test_hiding_method_keeps_fixed_common_support(tmp_path):
    paths = fixture(tmp_path, unsupported=True)
    snapshot = state(paths)
    result = data.view(snapshot, rows=data.detail(paths["database"], snapshot), methods=["xgb"])
    assert result["table"][0]["n"] == 3


def test_own_support_deltas_disabled(tmp_path):
    paths = fixture(tmp_path, unsupported=True)
    snapshot = state(paths)
    result = data.view(
        snapshot, rows=data.detail(paths["database"], snapshot), support="own_supported"
    )
    assert {r["n"] for r in result["table"]} == {3, 4}
    assert all(r["score_minus_baseline"] is None for r in result["table"])


def test_zero_baseline(tmp_path):
    paths = fixture(tmp_path, zero_baseline=True)
    assert all(
        r["improvement_percent"] is None
        for r in data.view(state(paths), baseline="seasonal_naive_7")["table"]
    )


def test_filters_without_details_refused(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="linked"):
        data.view(state(paths), series=['["a"]'])


def test_readonly_sql(tmp_path):
    paths = fixture(tmp_path)
    conn = data.connect(paths["database"])
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE forbidden(x)")
    finally:
        conn.close()


def test_inputs_unchanged(tmp_path):
    paths = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    snapshot = state(paths)
    data.view(snapshot, rows=data.detail(paths["database"], snapshot))
    assert data.model_settings(paths["database"], snapshot)["parameters"]["nested_fixture"] == [
        3,
        4,
    ]
    assert all(Path(p).read_bytes() == old for p, old in before.items())


def test_missing_database_not_created(tmp_path):
    missing = tmp_path / "absent.sqlite"
    assert data.inventory(missing)[0] == [] and not missing.exists()


def test_empty_database_graceful(tmp_path):
    path = tmp_path / "empty.sqlite"
    path.touch()
    assert data.inventory(path)[0] == []


def test_invalid_report_id(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="registered"):
        data.snapshot(paths["database"], "../../other")


def test_report_tamper(tmp_path):
    paths = fixture(tmp_path)
    (paths["report_path"] / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        state(paths)


def test_score_tamper_summary_survives(tmp_path):
    paths = fixture(tmp_path)
    snapshot = state(paths)
    (paths["score_path"] / "scores.jsonl").write_text("bad")
    assert data.view(snapshot)["table"]
    with pytest.raises(ValueError, match="integrity"):
        data.detail(paths["database"], snapshot)


def test_detail_budget(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        data.detail(paths["database"], state(paths), 2)


def test_csv_metadata(tmp_path):
    paths = fixture(tmp_path)
    text = data.comparison_csv(data.view(state(paths)))
    assert "report_id" in text and "selected_series" in text and "xgb" in text


def test_actual_plotly_figures(tmp_path):
    paths = fixture(tmp_path)
    snapshot = state(paths)
    result = data.view(snapshot, rows=data.detail(paths["database"], snapshot))
    charts = figures(result, snapshot["methods"], '["a"]')
    assert len(charts["pooled"].data) == 3 and len(charts["actual"].data) == 4
    assert all(trace.mode == "markers" for trace in charts["actual"].data)
    for key in ["pooled", "origins", "series", "actual", "residuals"]:
        assert json.loads(charts[key].to_json())["layout"]
    before = json.dumps(result["table"])
    assert figures(result, snapshot["methods"], '["a"]', max_points=1)["notes"]
    assert json.dumps(result["table"]) == before


def test_largest_errors(tmp_path):
    paths = fixture(tmp_path)
    snapshot = state(paths)
    result = data.view(snapshot, rows=data.detail(paths["database"], snapshot), methods=["xgb"])
    assert data.largest_errors(result)[0]["absolute_error"] == 3


@pytest.mark.parametrize(
    "kwargs", [{"methods": []}, {"metric": "mape"}, {"support": "bad"}, {"origins": ["2099-01-01"]}]
)
def test_invalid_controls(tmp_path, kwargs):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError):
        data.view(state(paths), **kwargs)


def test_registry_hash_binding(tmp_path):
    paths = fixture(tmp_path)
    with sqlite3.connect(paths["database"]) as conn:
        conn.execute("UPDATE temporal_acceptance_artifacts SET manifest_sha256='wrong'")
    with pytest.raises(ValueError, match="binding"):
        state(paths)


def test_missing_scores_summary_still_loads(tmp_path):
    paths = fixture(tmp_path)
    for file in paths["score_path"].iterdir():
        file.unlink()
    assert data.view(state(paths))["table"]


def test_unknown_model_settings_do_not_run_inference(tmp_path):
    paths = fixture(tmp_path)
    with sqlite3.connect(paths["database"]) as conn:
        conn.execute("DELETE FROM temporal_xgb_artifacts WHERE kind='models'")
    assert data.view(state(paths))["table"]
    with pytest.raises(ValueError):
        data.model_settings(paths["database"], state(paths))


def test_no_execution_imports():
    import ast

    forbidden = (
        "ml_template.models",
        "ml_template.training",
        "ml_template.evaluation",
        "ml_template.forecasting.temporal_xgb_io",
    )
    for node in ast.walk(ast.parse(Path(data.__file__).read_text())):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(forbidden)


def test_unknown_registry_table_refused(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="registry"):
        data.registry(paths["database"], "model_runs", "report")


def test_csv_records_actual_filters(tmp_path):
    paths = fixture(tmp_path)
    snapshot = state(paths)
    result = data.view(snapshot, rows=data.detail(paths["database"], snapshot), series=['["a"]'])
    assert json.loads(result["table"][0]["selected_series"]) == ['["a"]']
