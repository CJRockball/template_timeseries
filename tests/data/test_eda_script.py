"""EDA tests using the 28-row fixture and an isolated SQLite database.

Run: python -m pytest tests/data/test_eda_script.py -q
These tests exercise extraction and EDA, not the ingestion implementation.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine

from ml_template.scripts import eda_script as eda

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "ai4i_small.csv"

VERSION = "test_ai4i_small"


@pytest.fixture
def config():
    """Self-contained contract for ai4i_small.csv, not the active project."""
    names = {
        "UDI": "udi",
        "Product ID": "product_id",
        "Type": "type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
        **{c: c.lower() for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
    }
    outcomes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    return {
        "data": {
            "entity_id": "UDI",
            "target": "Machine failure",
            "task": "binary_classification",
            "positive_label": 1,
        },
        "bronze_source_contract": {
            "rows": 27,
            "columns": len(names),
            "col_names": list(names),
            "roles": {
                "id": ["UDI", "Product ID"],
                "feature": features,
                "target": ["Machine failure"],
                "outcome": outcomes,
            },
        },
        "validation": {
            "require_unique_entity_id": True,
            "max_missing_rate": 0.0,
            "allowed_target_values": [0, 1],
        },
        "silver_eda_contract": {
            "canonical_names": names,
            "canonical_dtypes": {
                "udi": "int64",
                "product_id": "object",
                "type": "object",
                "air_temperature_k": "float64",
                "process_temperature_k": "float64",
                "rotational_speed_rpm": "int64",
                "torque_nm": "float64",
                "tool_wear_min": "int64",
                "machine_failure": "int8",
                **{c.lower(): "int8" for c in outcomes},
            },
            "value_mappings": {"machine_failure": {"0": 0, "1": 1}},
            "missingness": {"default_max_rate": 0.0},
        },
    }


@pytest.fixture
def bronze(tmp_path, monkeypatch):
    raw = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    assert len(raw) == 27, "The fixture size changed; review expectations."
    raw.insert(0, "source_row_number", range(1, len(raw) + 1))
    raw.insert(0, "dataset_version_id", VERSION)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    raw.to_sql("raw_records", engine, index=False)
    monkeypatch.setattr(eda, "get_engine", lambda: engine)
    try:
        yield raw
    finally:
        engine.dispose()


@pytest.fixture
def canonical(bronze, config):
    return eda.load_and_canonicalize(VERSION, config)


def test_load_count_names_dtypes_and_roles(canonical, config):
    df, roles = canonical
    contract = config["silver_eda_contract"]
    assert len(df) == 27
    assert set(df["dataset_version_id"]) == {VERSION}
    assert df["source_row_number"].tolist() == list(range(1, 28))
    for col, dtype in contract["canonical_dtypes"].items():
        assert str(df[col].dtype) == dtype
    for raw, name in contract["canonical_names"].items():
        assert name in df.columns
        if raw != name:
            assert raw not in df.columns
    assert set(roles["feature"]).isdisjoint(roles["target"] + roles["outcome"])
    assert set(roles["feature"]).isdisjoint(roles["id"])


def test_duplicate_count_matches_source_records(canonical, bronze):
    df, _ = canonical
    source_cols = bronze.columns.difference(["dataset_version_id", "source_row_number"])
    expected = int(bronze[source_cols].duplicated().sum())
    assert eda.check_duplicates(df) == expected


@pytest.mark.xfail(
    strict=True,
    reason="Remove this marker after check_duplicates excludes lineage columns.",
)
def test_duplicate_detection_ignores_lineage(canonical):
    df, _ = canonical
    duplicate = df.iloc[[0]].copy()
    duplicate["source_row_number"] = int(df["source_row_number"].max()) + 1
    expanded = pd.concat([df, duplicate], ignore_index=True)
    assert eda.check_duplicates(expanded) == eda.check_duplicates(df) + 1


def test_target_counts_match_fixture(canonical):
    df, roles = canonical
    columns = roles["target"] + roles["outcome"]
    summary = eda.summarize_targets(df, columns)
    for col in columns:
        actual = summary[col]
        assert actual.get(0, 0) == int(df[col].eq(0).sum())
        assert actual.get(1, 0) == int(df[col].eq(1).sum())
        assert actual["proportion_%"] == round(df[col].eq(1).mean() * 100, 2)


def test_iqr_outlier_rows_are_actual_flags(canonical):
    df, roles = canonical
    numeric = [c for c in roles["feature"] if pd.api.types.is_numeric_dtype(df[c])]
    result = eda.detect_iqr_outliers(df, numeric)
    for col in numeric:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        assert result[col]["n_outliers"] == int(mask.sum())
        assert result[col]["outlier_rows"] == df.index[mask].tolist()


def test_run_eda_writes_machine_readable_report(bronze, config, tmp_path):
    test_config = tmp_path / "fixture.yaml"
    test_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "eda_output"
    report = eda.run_eda(VERSION, test_config, output)
    assert report.row_count == 27
    assert report.dataset_version_id == VERSION
    path = output / f"eda_report_{VERSION}.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["row_count"] == 27
    assert saved["dataset_version_id"] == VERSION
    assert saved["target_outcome_summary"]
    assert saved["numeric_summary"]
    assert saved["mutual_information"]
    assert (
        saved["proposed_silver_contract"]["canonical_names"]
        == (config["silver_eda_contract"]["canonical_names"])
    )
    for name in (
        "target_outcome_distribution.png",
        "numeric_by_target_violin.png",
        "spearman.png",
    ):
        assert (output / name).stat().st_size > 0
