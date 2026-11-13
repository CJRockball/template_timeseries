import json

import pandas as pd
import pytest
import yaml

from ml_template.preprocessing.smoke_nn import run_smoke


def smoke_config(tmp_path, branch="mixed", **kwargs):
    config = {
        "numeric": ["x"] if branch != "categorical" else [],
        "categorical": ["c"] if branch != "numeric" else [],
    }
    pre = tmp_path / "pre.yaml"
    pre.write_text(yaml.safe_dump(config))
    raw = {
        "preprocessor": str(pre),
        "source": {"kind": "synthetic", "rows": 32},
        "sample_rows": 16,
        "seed": 42,
    }
    raw.update(kwargs)
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("branch", ["numeric", "categorical", "mixed"])
def test_smoke_branch_and_persistence(tmp_path, branch):
    path = smoke_config(tmp_path, branch)
    destination = tmp_path / "smoke_output"
    report = run_smoke(path, destination)
    assert report["status"] == "passed"
    assert report["model_trained"] is False
    assert report["device"] == "cpu"
    assert json.loads((destination / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        run_smoke(path, destination)


def test_smoke_deterministic(tmp_path):
    path = smoke_config(tmp_path)
    assert run_smoke(path) == run_smoke(path)


@pytest.mark.parametrize(
    "raw",
    [
        {"sample_rows": 3},
        {"seed": True},
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"validation_fraction": True},
        {"surprise": 3},
    ],
)
def test_smoke_rejects_invalid(tmp_path, raw):
    with pytest.raises(ValueError):
        run_smoke(smoke_config(tmp_path, **raw))


def test_development_source_does_not_fit_holdout_or_read_target(tmp_path):
    pytest.importorskip("pyarrow")
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "x": [1.0, 2.0, 3.0, 4.0, 99999.0],
            "c": ["A", "A", "B", "B", "HOLDOUT"],
            "target": ["do not read"] * 5,
        }
    )
    source = tmp_path / "data.parquet"
    data.to_parquet(source, index=False)
    assignment = tmp_path / "assignment.parquet"
    pd.DataFrame({"id": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"]}).to_parquet(
        assignment
    )
    path = smoke_config(
        tmp_path,
        source={
            "kind": "development",
            "data": str(source),
            "assignments": str(assignment),
            "key": "id",
        },
    )
    output = tmp_path / "report"
    result = run_smoke(path, output)
    state = json.loads((output / "preprocessor.json").read_text())
    assert result["n_fit"] + result["n_validation"] == 4
    assert state["numeric_state"]["x"]["center"] < 5
    assert "HOLDOUT" not in json.dumps(state)
    assert result["holdout_labels_loaded"] is False
