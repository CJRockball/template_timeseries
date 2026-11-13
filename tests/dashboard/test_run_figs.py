import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import run_figs as rf


@pytest.fixture
def curves():
    rows = []
    for k in (0, 1):
        for r in range(1, 11):
            rows.append((k, r, "train", "logloss", 1.0 / r, 5))
            rows.append((k, r, "val", "logloss", 1.0 / r + 0.01 * r, 5))
    return pd.DataFrame(rows, columns=["fold", "round", "dataset", "metric", "value", "best_round"])


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    p = np.clip(0.1 * y + rng.random(400) * 0.5, 0, 1)
    return y, p


def test_missing_data_gives_placeholder():
    for fig in (
        rf.learning_curve_fig(None),
        rf.gap_fig(None),
        rf.importance_fig(None),
        rf.fold_metric_fig(None, "pr_auc"),
        rf.cost_fig(None, "r1"),
    ):
        assert len(fig.data) == 0 and fig.layout.annotations


def test_learning_curve_panels_and_overlay(curves):
    panels = rf.learning_curve_fig(curves)
    assert len(panels.data) == 4 and sum(t.showlegend for t in panels.data) == 2
    overlay = rf.learning_curve_fig(curves, overlay=True)
    assert len(overlay.data) == 4


def test_gap_is_val_minus_train(curves):
    fig = rf.gap_fig(curves)
    assert len(fig.data) == 2
    assert fig.data[0].y[0] == pytest.approx(0.01)


def test_fold_metric_skips_pooled_bar():
    m = pd.DataFrame(
        {"split": "cv", "fold": [-1, 0, 1], "metric": "pr_auc", "value": [0.8, 0.7, 0.9]}
    )
    fig = rf.fold_metric_fig(m, "pr_auc")
    assert list(fig.data[0].x) == ["fold 0", "fold 1"]
    assert rf.fold_metric_fig(m, "other").layout.annotations


def test_importance_sorted_and_limited():
    imp = pd.DataFrame(
        {
            "method": "perm",
            "feature": list("abcd"),
            "value": [0.1, 0.4, 0.2, 0.3],
            "std": [0.01, 0.02, np.nan, 0.01],
        }
    )
    fig = rf.importance_fig(imp, method="perm", top_n=3)
    assert list(fig.data[0].y) == ["c", "d", "b"]


def test_curves_from_scores(scored):
    y, p = scored
    assert len(rf.pr_curve_fig(y, p).data) == 1
    assert len(rf.reliability_fig(y, p).data) == 2
    assert len(rf.score_hist_fig(y, p, threshold=0.3).data) == 2
    assert rf.pr_curve_fig(np.zeros(5), np.zeros(5)).layout.annotations


def test_cost_highlights_selected_run():
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "duration_s": [10.0, 20.0],
            "primary_value": [0.7, 0.8],
            "primary_metric": "pr_auc",
        }
    )
    fig = rf.cost_fig(runs, "b")
    assert list(fig.data[1].text) == ["b"]
