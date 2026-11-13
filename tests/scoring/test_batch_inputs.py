from types import SimpleNamespace

import pandas as pd
import pytest

from ml_template.scoring.batch import check_batch_definition, read_batch
from ml_template.scoring.input_contract import frame_schema


def test_csv_string_keys(tmp_path):
    path = tmp_path / "batch.csv"
    path.write_text("id,v\n001,2\n002,3\n")
    contract = {
        "key_col": "id",
        "key_kind": "str",
        "source_schema": frame_schema(pd.DataFrame({"v": [2, 3]})),
    }
    assert read_batch(path, contract).id.tolist() == ["001", "002"]


def test_csv_labels_rejected(tmp_path):
    path = tmp_path / "batch.csv"
    path.write_text("id,v,y_true\n1,2,0\n")
    contract = {
        "key_col": "id",
        "key_kind": "int",
        "source_schema": frame_schema(pd.DataFrame({"v": [2]})),
    }
    with pytest.raises(ValueError, match="extra"):
        read_batch(path, contract)


def test_budget_batch_definition():
    predictor = SimpleNamespace(
        policy=SimpleNamespace(kind="alert_rate", batch_definition="entire scoring batch")
    )
    with pytest.raises(ValueError, match="acknowledgement"):
        check_batch_definition(predictor, None)
    with pytest.raises(ValueError, match="differs"):
        check_batch_definition(predictor, "per chunk")
    check_batch_definition(predictor, "entire scoring batch")


def test_no_policy_needs_no_acknowledgement():
    predictor = SimpleNamespace(policy=SimpleNamespace(kind="none", batch_definition="unused"))
    check_batch_definition(predictor, None)
