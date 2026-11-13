import numpy as np
import pandas as pd

from ml_template.evaluation.error_sample import error_sample


def test_lists_lowest_scored_positives_and_highest_scored_negatives():
    idx = pd.Index(np.arange(10), name="key")
    X = pd.DataFrame({"a": np.arange(10) * 10.0, "grade": list("abcdefghij")}, index=idx)
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=idx)
    oof = pd.DataFrame(
        {
            "key": np.arange(10),
            "fold": 0,
            "p_oof": [0.9, 0.1, 0.2, 0.8, 0.7, 0.6, 0.0, 0.1, 0.2, 0.3],
        }
    )
    out = error_sample(oof, X, y, "key", size=2)
    missed = out[out["error_kind"] == "missed_positive"]
    alarms = out[out["error_kind"] == "false_alarm"]
    assert list(missed["key"]) == [1, 2]  # the two lowest-scored positives
    assert list(alarms["key"]) == [3, 4]  # the two highest-scored negatives
    assert set(out.columns) >= {"key", "fold", "p_oof", "y_true", "error_kind", "a", "grade"}
    assert list(out["a"]) == [10.0, 20.0, 30.0, 40.0]  # features joined by key


def test_fewer_rows_than_size_is_fine():
    idx = pd.Index([0, 1, 2], name="key")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    y = pd.Series([1, 0, 0], index=idx)
    oof = pd.DataFrame({"key": [0, 1, 2], "fold": 0, "p_oof": [0.2, 0.9, 0.1]})
    out = error_sample(oof, X, y, "key", size=5)
    assert len(out) == 3


def test_feature_name_clash_is_rejected():
    idx = pd.Index([0, 1, 2], name="key")
    X = pd.DataFrame({"p_oof": [1.0, 2.0, 3.0]}, index=idx)
    y = pd.Series([1, 0, 0], index=idx)
    oof = pd.DataFrame({"key": [0, 1, 2], "fold": 0, "p_oof": [0.2, 0.9, 0.1]})
    import pytest

    with pytest.raises(ValueError, match="clash"):
        error_sample(oof, X, y, "key", size=1)
