import math

import numpy as np
import pytest

from ml_template.evaluation import metrics as m


def test_recall_and_precision_at_alert_rate():
    y = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.4, 0.05, 0.06, 0.07]
    assert m.recall_at_alert_rate(y, s, 0.2) == pytest.approx(0.5)
    assert m.precision_at_alert_rate(y, s, 0.2) == pytest.approx(0.5)
    assert m.recall_at_alert_rate(y, s, 0.4) == pytest.approx(1.0)


def test_alert_rate_rounds_up_and_floors_at_one_row():
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9] + [0.1] * 9
    assert m.recall_at_alert_rate(y, s, 0.01) == 1.0  # k = 1, not 0
    assert m.precision_at_alert_rate(y, s, 0.15) == pytest.approx(0.5)  # k = 2


def test_tie_break_is_deterministic():
    y = [1, 0, 0, 0]
    s = [0.5, 0.5, 0.5, 0.5]
    assert m.recall_at_alert_rate(y, s, 0.25) == 1.0
    assert m.recall_at_alert_rate(y, s, 0.25) == m.recall_at_alert_rate(y, s, 0.25)


def test_pr_auc_perfect_and_single_class():
    assert m.pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert math.isnan(m.pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert math.isnan(m.roc_auc([1, 1], [0.1, 0.2]))
    assert math.isnan(m.recall_at_alert_rate([0, 0, 0], [0.1, 0.2, 0.3], 0.5))


def test_input_validation():
    with pytest.raises(ValueError):
        m.pr_auc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.pr_auc([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        m.pr_auc([0, 1, 1], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.pr_auc([], [])
    with pytest.raises(ValueError):
        m.recall_at_alert_rate([0, 1], [0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        m.recall_at_alert_rate([0, 1], [0.1, 0.2], 1.5)
    with pytest.raises(ValueError):
        m.brier([0, 1], [0.1, 1.5])


def test_keys_and_resolved_keys():
    assert m.alert_label(0.05) == "5pct"
    assert m.alert_label(0.125) == "12.5pct"
    names = ["pr_auc", "recall_at_alert_rate", "precision_at_alert_rate"]
    assert m.resolved_keys(names, [0.05, 0.1]) == [
        "pr_auc",
        "recall_at_5pct",
        "recall_at_10pct",
        "precision_at_5pct",
        "precision_at_10pct",
    ]
    with pytest.raises(ValueError):
        m.resolved_keys(["recall_at_alert_rate"], [])
    with pytest.raises(ValueError):
        m.resolved_keys(["nope"])


def test_compute_metrics_matches_resolved_keys():
    y = [0, 1, 0, 1, 0, 0, 1, 0]
    s = [0.1, 0.9, 0.2, 0.8, 0.3, 0.15, 0.7, 0.05]
    names = ["pr_auc", "roc_auc", "brier", "recall_at_alert_rate", "precision_at_alert_rate"]
    out = m.compute_metrics(y, s, names, [0.25, 0.5])
    assert list(out) == m.resolved_keys(names, [0.25, 0.5])
    assert out["recall_at_25pct"] == pytest.approx(2 / 3)


def test_long_form_rows():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    s = rng.random(40)
    folds = np.repeat([0, 1], 20)
    rows = m.long_form_metrics(y, s, ["pr_auc"], split="cv", folds=folds)
    assert {(r["split"], r["fold"]) for r in rows} == {("cv", -1), ("cv", 0), ("cv", 1)}
    hold = m.long_form_metrics(y, s, ["pr_auc"], split="holdout")
    assert [(r["fold"], r["metric"]) for r in hold] == [(-1, "pr_auc")]
    with pytest.raises(ValueError):
        m.long_form_metrics(y, s, ["pr_auc"], split="cv", folds=[0, 1])


def test_with_standard_adds_but_never_removes():
    names, rates = m.with_standard(None, None)
    assert names == list(m.STANDARD_METRICS) and rates == [0.05, 0.10]
    names, rates = m.with_standard(["roc_auc"], [0.01, 0.05])
    assert names == list(m.STANDARD_METRICS)  # roc_auc was already standard
    assert rates == [0.01, 0.05, 0.10]
    assert m.with_standard(["extra"], [])[0][-1] == "extra"


def test_label_summary_rows():
    rows = m.label_summary_rows([0, 0, 0, 1], split="cv")
    values = {r["metric"]: r["value"] for r in rows}
    assert values == {"n_rows": 4.0, "n_positives": 1.0, "base_rate": 0.25}
    assert all(r["split"] == "cv" and r["fold"] == -1 for r in rows)


def test_fold_stability_rows_mean_and_sample_std():
    rows = [
        {"split": "cv", "fold": -1, "metric": "pr_auc", "value": 0.5},
        {"split": "cv", "fold": 0, "metric": "pr_auc", "value": 0.4},
        {"split": "cv", "fold": 1, "metric": "pr_auc", "value": 0.6},
        {"split": "cv", "fold": 2, "metric": "pr_auc", "value": float("nan")},
    ]
    out = {r["metric"]: r["value"] for r in m.fold_stability_rows(rows, ["pr_auc"])}
    assert out["pr_auc__fold_mean"] == pytest.approx(0.5)
    assert out["pr_auc__fold_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert m.fold_stability_rows(rows[:2], ["pr_auc"]) == []  # one valid fold is not enough
