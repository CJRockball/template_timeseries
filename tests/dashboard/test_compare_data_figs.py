import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard import compare_data as cd
from ml_template.dashboard import compare_figs as cf


def _predictions():
    a = pd.DataFrame(
        {
            "id": [3, 1, 4, 2],
            "fold": [1, 0, 1, 0],
            "y_true": [0, 1, 1, 0],
            "p_oof": [0.1, 0.9, 0.2, 0.8],
        }
    )
    b = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fold": [0, 0, 1, 1],
            "y_true": [1, 0, 0, 1],
            "p_oof": [0.8, 0.1, 0.2, 0.9],
        }
    )
    return {"a": a, "b": b}


def _headers():
    return pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "silver_dataset_version_id": "s",
            "split_id": "sp",
            "primary_metric": "pr_auc",
            "status": "succeeded",
            "duration_s": [10.0, 20.0],
            "primary_value": [0.7, 0.9],
        }
    )


def _metrics():
    return pd.DataFrame(
        [
            {"run_id": rid, "split": split, "fold": fold, "metric": "pr_auc", "value": value}
            for rid, split, fold, value in [
                ("a", "cv", -1, 0.7),
                ("b", "cv", -1, 0.9),
                ("a", "cv", 0, 0.6),
                ("b", "cv", 0, 0.8),
                ("a", "holdout", -1, 0.99),
            ]
        ]
    )


def test_selection_order_and_minimum():
    assert cd.selected_runs(_headers(), ["b", "a"]).run_id.tolist() == ["b", "a"]
    for ids in (["a"], ["a", "a"], ["a", "missing"]):
        with pytest.raises(ValueError):
            cd.selected_runs(_headers(), ids)


def test_cohort_and_status_guards():
    for column, value in [
        ("split_id", "other"),
        ("status", "failed"),
        ("primary_metric", "brier"),
        ("silver_dataset_version_id", "other"),
    ]:
        headers = _headers()
        headers.loc[1, column] = value
        with pytest.raises(ValueError):
            cd.selected_runs(headers, ["a", "b"])


def test_alignment_is_keyed_not_positional():
    aligned = cd.align_predictions(_predictions(), "id")
    assert aligned["a"].id.tolist() == [1, 2, 3, 4]
    assert aligned["a"].p_oof.tolist() == [0.9, 0.8, 0.1, 0.2]


@pytest.mark.parametrize("change", ["duplicate", "missing_key", "label", "fold", "nan", "range"])
def test_bad_predictions_rejected(change):
    tables = _predictions()
    b = tables["b"].copy()
    if change == "duplicate":
        b.loc[1, "id"] = 1
    elif change == "missing_key":
        b = b.iloc[:3]
    elif change == "label":
        b.loc[0, "y_true"] = 0
    elif change == "fold":
        b.loc[0, "fold"] = 1
    elif change == "nan":
        b.loc[0, "p_oof"] = np.nan
    else:
        b.loc[0, "p_oof"] = 1.1
    tables["b"] = b
    with pytest.raises(ValueError):
        cd.align_predictions(tables, "id")


def test_holdout_or_partial_development_rows_rejected():
    assignments = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"], "fold": [0, 0, 1, 1, -1]}
    )
    assert len(cd.align_predictions(_predictions(), "id", assignments)) == 2
    assignments.loc[3, "partition"] = "holdout"
    with pytest.raises(ValueError, match="stored development"):
        cd.align_predictions(_predictions(), "id", assignments)


def test_label_join_has_no_silent_drops():
    oof = _predictions()["a"].drop(columns="y_true")
    labels = pd.Series([1, 0, 0], index=pd.Index([1, 2, 3], name="id"))
    attached = cd.attach_labels(oof, labels, "id")
    assert len(attached) == 4
    with pytest.raises(ValueError, match="labels"):
        cd.align_predictions({"a": attached}, "id")


def test_metric_table_excludes_holdout():
    table = cd.metric_table(_metrics(), ["a", "b"])
    assert table.iloc[0]["a"] == 0.7


def test_parameter_diff_and_missing_values():
    table = cd.parameter_diff(
        {"a": {"seed": "1", "depth": "3"}, "b": {"seed": "2", "depth": "3", "new": "yes"}}
    )
    assert set(table.parameter) == {"seed", "new"}
    assert table.set_index("parameter").loc["new", "a"] == "<not recorded>"


def test_feature_membership_missing_is_not_false():
    table = cd.feature_membership({"a": ["x"], "b": ["x", "y"], "c": None})
    assert not bool(table.set_index("feature").loc["y", "a"])
    assert table["c"].isna().all()


def test_cutoff_errors_and_complementarity():
    tables = cd.align_predictions(_predictions(), "id")
    operating, errors = cd.prediction_tables(tables, "a", "threshold", cutoff=0.5)
    a = operating.set_index("run_id").loc["a"]
    b = operating.set_index("run_id").loc["b"]
    assert a["fn"] == 1 and b["fn"] == 0
    assert set(errors.id) == {2, 4}
    assert not errors.missed_by_all.any()


def test_budget_uses_equal_counts_with_stable_ties():
    tables = cd.align_predictions(_predictions(), "id")
    for table in tables.values():
        table["p_oof"] = 0.5
    summary, _ = cd.prediction_tables(tables, "a", "budget", alert_rate=0.25)
    assert summary.flagged.tolist() == [1, 1]
    assert summary.tp.tolist() == [1, 1]


def test_invalid_operating_controls():
    tables = cd.align_predictions(_predictions(), "id")
    with pytest.raises(ValueError):
        cd.prediction_tables(tables, "missing", "threshold")
    with pytest.raises(ValueError):
        cd.prediction_tables(tables, "a", "budget", alert_rate=0)


def test_pr_and_reliability_overlays():
    tables = cd.align_predictions(_predictions(), "id")
    assert len(cf.pr_overlay(tables).data) == 2
    assert len(cf.reliability_overlay(tables).data) == 3
    tables["a"]["p_oof"] = 0.5
    fig = cf.reliability_overlay(tables)
    assert len(fig.data[1].x) == 1


def test_delta_and_fold_plots():
    fig = cf.metric_deltas(_metrics(), "a", "pr_auc", higher=True)
    assert fig.data[0].y[0] == pytest.approx(0.2)
    assert len(cf.fold_overlay(_metrics(), "pr_auc").data) == 2


def test_missing_importance_remains_blank():
    importance = pd.DataFrame(
        {"run_id": ["a", "b"], "method": "permutation", "feature": ["x", "y"], "value": [0.1, 0.2]}
    )
    fig = cf.importance_heatmap(importance, ["a", "b"], "permutation")
    assert pd.isna(fig.data[0].z).sum() == 2


def test_learning_curves_filter_metric_and_fold():
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 1],
            "round": [1, 2, 1],
            "dataset": "val",
            "metric": "logloss",
            "value": [0.6, 0.5, 0.4],
        }
    )
    fig = cf.learning_overlay({"a": frame, "b": None}, "logloss", 0)
    assert len(fig.data) == 1 and list(fig.data[0].x) == [1, 2]
    assert cf.learning_overlay({}, None, None).layout.annotations


def test_cost_and_missing_prediction_placeholders():
    assert len(cf.cost_plot(_headers()).data) == 2
    assert cf.pr_overlay({}).layout.annotations
    assert cf.reliability_overlay({}).layout.annotations


def test_all_runs_sharing_same_incomplete_subset_still_rejected():
    tables = {rid: frame[frame["id"] != 4] for rid, frame in _predictions().items()}
    assignments = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"], "fold": [0, 0, 1, 1, -1]}
    )
    with pytest.raises(ValueError, match="stored development"):
        cd.align_predictions(tables, "id", assignments)


def test_positive_missed_by_every_run_is_in_error_table():
    tables = cd.align_predictions(_predictions(), "id")
    tables["b"].loc[tables["b"]["id"] == 4, "p_oof"] = 0.1
    _, errors = cd.prediction_tables(tables, "a", "threshold", cutoff=0.5)
    assert errors.set_index("id").loc[4, "missed_by_all"]
