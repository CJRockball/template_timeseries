import json

import pandas as pd
import pytest
import yaml

from ml_template.preprocessing.smoke_nn import run_smoke


def test_csv_development_selection_with_assignment_io_fixture(tmp_path, monkeypatch):
    pre = tmp_path / "pre.yaml"
    pre.write_text("numeric: [x]\ncategorical: [c]\n")
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "id": [10, 20, 30, 40, 50],
            "x": [1.0, 2.0, 3.0, 4.0, 99999.0],
            "c": ["A", "B", "A", "B", "HOLDOUT"],
            "target": ["FORBIDDEN"] * 5,
        }
    ).to_csv(source, index=False)
    assignments = pd.DataFrame({"id": [10, 20, 30, 40, 50], "partition": ["dev"] * 4 + ["holdout"]})
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: assignments[columns].copy())
    path = tmp_path / "smoke.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "preprocessor": str(pre),
                "source": {
                    "kind": "development",
                    "data": str(source),
                    "assignments": "assignment.parquet",
                    "key": "id",
                },
                "sample_rows": 16,
            }
        )
    )
    destination = tmp_path / "report"
    report = run_smoke(path, destination)
    state = json.loads((destination / "preprocessor.json").read_text())
    assert report["n_fit"] + report["n_validation"] == 4
    assert state["numeric_state"]["x"]["center"] < 5
    assert "HOLDOUT" not in json.dumps(state)
    assert "FORBIDDEN" not in json.dumps(state)


def test_development_source_missing_keys_fails(tmp_path, monkeypatch):
    pre = tmp_path / "pre.yaml"
    pre.write_text("numeric: [x]\n")
    source = tmp_path / "source.csv"
    pd.DataFrame({"id": [1, 2], "x": [1, 2]}).to_csv(source, index=False)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, columns: pd.DataFrame({"id": [9], "partition": ["dev"]})[columns],
    )
    path = tmp_path / "smoke.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "preprocessor": str(pre),
                "source": {
                    "kind": "development",
                    "data": str(source),
                    "assignments": "a.parquet",
                    "key": "id",
                },
            }
        )
    )
    with pytest.raises(ValueError, match="development keys missing"):
        run_smoke(path)
