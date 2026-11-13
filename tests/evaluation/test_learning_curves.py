import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.learning_curves import (
    diagnostic_rounds,
    learning_curves_for_run,
    plot_learning_curves,
)


class FakeAdapter:
    max_rounds = 100

    def __init__(self, calls):
        self.calls = calls

    def learning_curve(self, X_train, y_train, X_val, y_val, n_rounds, metric):
        self.calls.append((sorted(X_train.index), sorted(X_val.index), n_rounds, metric))
        r = np.arange(1, n_rounds + 1)
        train = pd.DataFrame({"round": r, "dataset": "train", "metric": metric, "value": 1 / r})
        val = pd.DataFrame({"round": r, "dataset": "val", "metric": metric, "value": 1 / r + 0.1})
        return pd.concat([train, val], ignore_index=True)


@pytest.fixture
def setup():
    keys = np.arange(1, 13)
    X = pd.DataFrame({"a": np.arange(12.0)}, index=pd.Index(keys, name="udi"))
    y = pd.Series([0, 1] * 6, index=X.index)
    fold = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1]
    assignments = pd.DataFrame(
        {
            "udi": keys,
            "partition": ["dev"] * 10 + ["holdout"] * 2,
            "fold": fold,
        }
    )
    folds = pd.DataFrame({"fold": [0, 1], "best_round": [5, 8]})
    calls = []
    return X, y, assignments, folds, calls, (lambda: FakeAdapter(calls))


def test_table_has_one_curve_per_fold_with_default_length(setup):
    X, y, assignments, folds, calls, factory = setup
    table = learning_curves_for_run(factory, X, y, assignments, "udi", folds, metric="logloss")
    assert list(table.columns) == ["fold", "round", "dataset", "metric", "value", "best_round"]
    assert set(table["dataset"]) == {"train", "val"} and set(table["metric"]) == {"logloss"}
    rounds = table.groupby("fold")["round"].max().to_dict()
    assert rounds == {0: 15, 1: 18}  # max(2 x best, best + 10)
    assert table.groupby("fold")["best_round"].first().to_dict() == {0: 5, 1: 8}


def test_only_development_rows_are_used_and_folds_are_disjoint(setup):
    X, y, assignments, folds, calls, factory = setup
    learning_curves_for_run(factory, X, y, assignments, "udi", folds, metric="logloss")
    (train0, val0, _, _), (train1, val1, _, _) = calls
    assert val0 == [1, 2, 3, 4, 5] and train0 == [6, 7, 8, 9, 10]
    assert val1 == [6, 7, 8, 9, 10] and train1 == [1, 2, 3, 4, 5]
    assert not ({11, 12} & set(train0 + val0 + train1 + val1))


def test_fixed_rounds_and_cap(setup):
    X, y, assignments, folds, calls, factory = setup
    fixed = learning_curves_for_run(
        factory, X, y, assignments, "udi", folds, metric="logloss", max_rounds=7
    )
    assert fixed.groupby("fold")["round"].max().tolist() == [7, 7]
    capped = learning_curves_for_run(
        factory, X, y, assignments, "udi", folds, metric="logloss", max_rounds=500
    )
    assert capped.groupby("fold")["round"].max().tolist() == [100, 100]


def test_diagnostic_rounds_rule():
    assert diagnostic_rounds(50, None, 2000) == 100
    assert diagnostic_rounds(3, None, 2000) == 13
    assert diagnostic_rounds(50, None, 60) == 60
    assert diagnostic_rounds(50, 30, 2000) == 30


def test_plot_is_written(setup, tmp_path):
    X, y, assignments, folds, calls, factory = setup
    table = learning_curves_for_run(factory, X, y, assignments, "udi", folds, metric="logloss")
    out = tmp_path / "curves.png"
    plot_learning_curves(table, out, title="run")
    assert out.exists() and out.stat().st_size > 0
