import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_template.evaluation.metrics import with_standard
from ml_template.training import cross_validation as cv
from ml_template.training.run_config import SplitConfig
from ml_template.training.split import make_split

KEY, TARGET = "udi", "machine_failure"
NAMES, RATES = with_standard([], [])


class FakeAdapter:
    """Linear model with a deterministic 'best round', plus a log of what it saw."""

    max_rounds = 100
    seen: list = []

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        FakeAdapter.seen += [("stop_fit", X_train.index), ("stop_val", X_val.index)]
        self.m = LogisticRegression(max_iter=300).fit(X_train, y_train)
        return 10 + len(X_train) % 7

    def refit(self, X, y, n_rounds):
        FakeAdapter.seen.append(("refit", X.index))
        self.n_rounds = n_rounds
        self.m = LogisticRegression(max_iter=300).fit(X, y)

    def predict_proba(self, X):
        FakeAdapter.seen.append(("predict", X.index))
        return self.m.predict_proba(X)[:, 1]

    def save(self, path):
        path.write_text("fake")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * rng.normal(size=n) > 1.6).astype(int)
    idx = pd.Index(np.arange(1, n + 1), name=KEY)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(y, index=idx, name=TARGET)
    assign = make_split(pd.DataFrame({KEY: idx, TARGET: y.to_numpy()}), SplitConfig(), KEY, TARGET)
    return X, y, assign


def _run(data, **kw):
    X, y, assign = data
    FakeAdapter.seen = []
    return cv.cross_validate(
        FakeAdapter,
        X,
        y,
        assign,
        KEY,
        validation_fraction=0.2,
        seed=1,
        metric_names=NAMES,
        alert_rates=RATES,
        **kw,
    )


def test_oof_covers_each_dev_row_once_and_never_holdout(data):
    _, _, assign = data
    res = _run(data)
    dev = set(assign.loc[assign["partition"] == "dev", KEY])
    assert len(res.oof) == len(dev) and set(res.oof[KEY]) == dev
    assert res.oof["p_oof"].between(0, 1).all()


def test_holdout_rows_are_never_seen_by_the_model(data):
    _, _, assign = data
    hold = set(assign.loc[assign["partition"] == "holdout", KEY])
    _run(data)
    for _, index in FakeAdapter.seen:
        assert not hold & set(index)


def test_validation_fold_is_not_used_for_early_stopping(data):
    res = _run(data)
    calls = FakeAdapter.seen
    assert len(calls) == 4 * len(res.folds)  # stop_fit, stop_val, refit, predict per fold
    for i in range(0, len(calls), 4):
        (_, stop_fit), (_, stop_val), (_, refit), (_, predict) = calls[i : i + 4]
        val, train = set(predict), set(refit)
        assert not val & train
        assert not val & (set(stop_fit) | set(stop_val))
        assert (set(stop_fit) | set(stop_val)) == train
        assert not set(stop_fit) & set(stop_val)


def test_fold_models_are_the_ones_that_predicted_each_fold(data):
    res = _run(data)
    assert sorted(res.fold_models) == sorted(res.fold_val_keys) == list(res.folds["fold"])
    assert len({id(m) for m in res.fold_models.values()}) == len(res.fold_models)
    for fold, keys in res.fold_val_keys.items():
        assert set(keys) == set(res.oof.loc[res.oof["fold"] == fold, KEY])
        X = data[0]
        p = res.fold_models[fold].predict_proba(X.loc[keys])
        np.testing.assert_allclose(p, res.oof.set_index(KEY).loc[keys, "p_oof"].to_numpy())


def test_metric_rows_and_final_rounds(data):
    res = _run(data)
    keys = {(r["split"], r["fold"], r["metric"]) for r in res.metric_rows}
    assert ("cv", -1, "pr_auc") in keys and ("cv", -1, "recall_at_5pct") in keys
    assert all(("cv", int(f), "best_round") in keys for f in res.folds["fold"])
    median = np.median(res.folds["best_round"])
    assert res.n_rounds_final == max(1, round(median))
    scaled = _run(data, refit_rounds_factor=1.5)
    assert scaled.n_rounds_final == max(1, round(median * 1.5))
    assert ("cv", -1, "n_rounds_final") in {
        (r["split"], r["fold"], r["metric"]) for r in scaled.metric_rows
    }


def test_standard_set_stability_and_label_summary_rows(data):
    X, y, assign = data
    res = _run(data)
    metrics = {(r["fold"], r["metric"]): r["value"] for r in res.metric_rows}
    for name in (
        "pr_auc",
        "roc_auc",
        "brier",
        "recall_at_5pct",
        "recall_at_10pct",
        "precision_at_5pct",
        "precision_at_10pct",
    ):
        assert (-1, name) in metrics, name
    per_fold = [metrics[(int(f), "pr_auc")] for f in res.folds["fold"]]
    assert metrics[(-1, "pr_auc__fold_mean")] == pytest.approx(np.mean(per_fold))
    assert metrics[(-1, "pr_auc__fold_std")] == pytest.approx(np.std(per_fold, ddof=1))
    dev_keys = assign.loc[assign["partition"] == "dev", KEY]
    assert metrics[(-1, "n_rows")] == len(dev_keys)
    assert metrics[(-1, "base_rate")] == pytest.approx(y.loc[dev_keys].mean())
    assert metrics[(-1, "n_positives")] == y.loc[dev_keys].sum()


def test_deterministic(data):
    a, b = _run(data), _run(data)
    pd.testing.assert_frame_equal(a.oof, b.oof)
    assert a.n_rounds_final == b.n_rounds_final


def test_fit_final_uses_dev_rows_only(data):
    X, y, assign = data
    FakeAdapter.seen = []
    adapter = cv.fit_final(FakeAdapter, X, y, assign, KEY, n_rounds=33)
    dev = set(assign.loc[assign["partition"] == "dev", KEY])
    ((_, index),) = [s for s in FakeAdapter.seen if s[0] == "refit"]
    assert set(index) == dev and adapter.n_rounds == 33


def test_misaligned_inputs_rejected(data):
    X, y, assign = data
    common = {"validation_fraction": 0.2, "seed": 1, "metric_names": NAMES, "alert_rates": RATES}
    with pytest.raises(ValueError, match="different rows"):
        cv.cross_validate(FakeAdapter, X.iloc[:-5], y.iloc[:-5], assign, KEY, **common)
    with pytest.raises(ValueError, match="same index"):
        cv.cross_validate(FakeAdapter, X, y.reset_index(drop=True), assign, KEY, **common)
