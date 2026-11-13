import sqlite3

import pytest

from ml_template.explainability.presentation_data import (
    list_explanations,
    load_explanation,
    selected_entity,
    table_records,
)


def test_verified_tables_align_by_keys(presentation_fixture):
    obj = presentation_fixture
    obj.tables["features.parquet"] = obj.tables["features.parquet"].iloc[::-1]
    view = obj.view()
    assert view.features.index.tolist() == [30, 10, 20]
    assert list_explanations(obj.database)[0]["explanation_id"] == "explain_fixture"
    key, _, table = selected_entity(view, 1)
    assert key == 10 and set(table.feature) == {"x", "kind"}
    assert isinstance(table_records(table), list)


def test_reader_never_writes(presentation_fixture):
    obj = presentation_fixture
    before = obj.database.read_bytes()
    obj.view()
    assert obj.database.read_bytes() == before


@pytest.mark.parametrize("position", [True, -1, 3, 0.5, "1"])
def test_invalid_selection(presentation_fixture, position):
    with pytest.raises(ValueError):
        selected_entity(presentation_fixture.view(), position)


def test_stale_registry_hash_rejected(presentation_fixture):
    obj = presentation_fixture
    with sqlite3.connect(obj.database) as conn:
        conn.execute("UPDATE explanation_runs SET content_sha256='wrong'")
    with pytest.raises(ValueError, match="fingerprint"):
        obj.view()


def test_unsuccessful_execution_rejected(presentation_fixture):
    obj = presentation_fixture
    with sqlite3.connect(obj.database) as conn:
        conn.execute("UPDATE explanation_runs SET status='failed'")
    with pytest.raises(ValueError, match="succeeded"):
        obj.view()


@pytest.mark.parametrize(
    "mutation",
    ["keys", "columns", "nonfinite", "additivity", "probability", "importance", "labels"],
)
def test_semantic_mismatch_rejected(presentation_fixture, mutation):
    obj = presentation_fixture
    if mutation == "keys":
        obj.tables["features.parquet"] = obj.tables["features.parquet"].iloc[:1]
    elif mutation == "columns":
        obj.tables["contributions.parquet"].columns = ["wrong", "kind"]
    elif mutation == "nonfinite":
        obj.tables["contributions.parquet"].iloc[0, 0] = float("nan")
    elif mutation == "additivity":
        obj.tables["contributions.parquet"].iloc[0, 0] += 1
    elif mutation == "probability":
        obj.tables["rows.parquet"].loc[30, "p_raw"] = 0.99
    elif mutation == "importance":
        obj.tables["global_summary.parquet"].loc[0, "mean_abs_contribution"] = 99
    elif mutation == "labels":
        obj.tables["rows.parquet"]["y_true"] = 0
    with pytest.raises(ValueError):
        obj.view()


def test_row_limit(presentation_fixture):
    with pytest.raises(ValueError, match="limit"):
        load_explanation(presentation_fixture.database, "explain_fixture", max_rows=1)


def test_missing_database_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(ValueError):
        list_explanations(path)
    assert not path.exists()
