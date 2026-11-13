import numpy as np
import pandas as pd
import pytest

from ml_template.scoring.input_contract import coerce_frame, frame_schema, validate_batch


def source():
    return pd.DataFrame(
        {
            "kind": pd.Categorical(["L", "M", "H"], categories=["L", "M", "H"]),
            "value": np.array([1, 2, 3], dtype="int64"),
            "speed": [1.0, 2.5, 3.0],
        }
    )


def test_subset_keeps_categorical_codes():
    frame = source()
    got = coerce_frame(frame.iloc[[2]].assign(kind="H"), frame_schema(frame))
    assert got.kind.cat.codes.tolist() == [2]
    assert got.kind.cat.categories.tolist() == ["L", "M", "H"]


def test_column_reordering():
    frame = source()
    pd.testing.assert_frame_equal(
        coerce_frame(frame.iloc[::-1][["speed", "value", "kind"]], frame_schema(frame)),
        frame.iloc[::-1],
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.drop(columns="value"),
        lambda f: f.assign(extra=1),
        lambda f: f.assign(value=1.5),
        lambda f: f.assign(value="2"),
        lambda f: f.assign(value=True),
        lambda f: f.assign(speed=np.inf),
        lambda f: f.assign(speed=np.nan),
        lambda f: f.assign(kind="new"),
        lambda f: f.iloc[:0],
    ],
)
def test_invalid_inputs(change):
    with pytest.raises(ValueError):
        coerce_frame(change(source()), frame_schema(source()))


@pytest.mark.parametrize("keys", [[1, 1, 3], [1, None, 3], ["1", "2", "3"], [True, False, True]])
def test_invalid_keys(keys):
    frame = source()
    contract = {"key_col": "id", "key_kind": "int", "source_schema": frame_schema(frame)}
    frame.insert(0, "id", keys)
    with pytest.raises(ValueError):
        validate_batch(frame, contract)


def test_labels_and_extras_rejected():
    frame = source()
    contract = {"key_col": "id", "key_kind": "int", "source_schema": frame_schema(frame)}
    frame.insert(0, "id", [1, 2, 3])
    with pytest.raises(ValueError, match="extra"):
        validate_batch(frame.assign(y_true=0), contract)


def test_keys_preserved():
    frame = source()
    contract = {"key_col": "id", "key_kind": "int", "source_schema": frame_schema(frame)}
    frame.insert(0, "id", [30, 10, 20])
    assert validate_batch(frame, contract).index.tolist() == [30, 10, 20]


def test_overflow():
    schema = frame_schema(pd.DataFrame({"v": np.array([1], dtype="int8")}))
    with pytest.raises(ValueError, match="range"):
        coerce_frame(pd.DataFrame({"v": [128]}), schema)


def test_string_vocabulary():
    schema = frame_schema(pd.DataFrame({"s": ["a", "b"]}))
    with pytest.raises(ValueError, match="unseen"):
        coerce_frame(pd.DataFrame({"s": ["c"]}), schema)


def test_duplicate_columns():
    with pytest.raises(ValueError):
        frame_schema(pd.DataFrame([[1, 2]], columns=["v", "v"]))
