"""Regression tests for run-weighted sweep parameter summaries."""

import math

import numpy as np
import pandas as pd
import pytest

from ml_template.dashboard.sweep_parameter_summary import parameter_summaries

P = "param::model.params."


def frame():
    return pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "experiment_run_id": ["launch"] * 6,
            "silver_dataset_version_id": ["silver"] * 6,
            "split_id": ["split"] * 6,
            "primary_metric": ["auc"] * 6,
            "status": ["succeeded"] * 6,
            "primary_value": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "params_recorded": [True] * 6,
            "seed": [42] * 6,
            P + "learning_rate": ["0.03", "0.03", "0.03", "0.1", "0.1", "0.1"],
            P + "colsample_bytree": ["0.6", "0.75", "0.9", "0.6", "0.75", "0.9"],
            P + "device": ["cuda"] * 6,
        }
    )


def row(table, parameter, value=None):
    selected = table[table["parameter"].eq("model.params." + parameter)]
    if value is not None:
        selected = selected[selected["value"].eq(value)]
    return selected.iloc[0]


def test_overall_pools_other_parameters_and_preserves_repeated_seeds():
    overall, _, meta = parameter_summaries(frame())
    result = row(overall, "learning_rate", "0.03")
    assert result["n_runs"] == 3
    assert result["mean"] == pytest.approx(0.8)
    assert result["std"] == pytest.approx(0.1)
    assert result["min"] == 0.7
    assert result["max"] == 0.9
    assert meta["eligible_runs"] == 6


def test_overall_sampling_group():
    overall, _, _ = parameter_summaries(frame())
    result = row(overall, "colsample_bytree", "0.6")
    assert result["n_runs"] == 2
    assert result["mean"] == pytest.approx(0.75)
    assert result["std"] == pytest.approx(0.3 / math.sqrt(2))


def test_top_three_and_baseline():
    _, top, meta = parameter_summaries(frame(), k=3)
    result = row(top, "learning_rate")
    assert result["mode_values"] == ["0.03"]
    assert result["mode_share_pct"] == 100
    assert result["mode_share_overall_pct"] == 50
    assert result["numeric_min"] == result["numeric_max"] == 0.03
    assert meta["selected_runs"] == 3


def test_tied_modes_have_separate_baselines():
    _, top, _ = parameter_summaries(frame(), k=3)
    result = row(top, "colsample_bytree")
    assert result["mode_values"] == ["0.6", "0.75", "0.9"]
    assert result["mode_share_pct"] == pytest.approx(100 / 3)
    assert pd.isna(result["mode_share_overall_pct"])
    assert len(result["mode_baselines_overall"]) == 3
    assert all(x["share_pct"] == pytest.approx(100 / 3) for x in result["mode_baselines_overall"])


def test_lower_is_better():
    overall, top, _ = parameter_summaries(frame(), k=3, higher_is_better=False)
    assert row(top, "learning_rate")["mode_values"] == ["0.1"]
    learning = overall[overall["parameter"].eq("model.params.learning_rate")]
    assert learning.iloc[0]["value"] == "0.1"


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf, "bad"])
def test_nonfinite_scores_excluded(bad):
    data = frame()
    data["primary_value"] = data["primary_value"].astype(object)
    data.loc[0, "primary_value"] = bad
    _, _, meta = parameter_summaries(data)
    assert meta["eligible_runs"] == 5
    assert meta["excluded_runs"] == 1


@pytest.mark.parametrize("status", ["failed", "running", "pending"])
def test_non_success_excluded(status):
    data = frame()
    data.loc[0, "status"] = status
    assert parameter_summaries(data)[2]["eligible_runs"] == 5


def test_fewer_than_ten():
    assert parameter_summaries(frame())[2]["selected_runs"] == 6


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_runs(count):
    overall, _, meta = parameter_summaries(frame().iloc[:count])
    assert meta["insufficient_runs"]
    if count:
        assert overall["std"].isna().all()


def test_empty_without_schema():
    overall, top, meta = parameter_summaries(pd.DataFrame())
    assert overall.empty and top.empty
    assert meta["eligible_runs"] == 0


def test_fixed_status():
    overall, top, _ = parameter_summaries(frame(), k=3)
    assert row(overall, "device", "cuda")["status"] == "Fixed throughout"
    assert row(top, "device")["status"] == "Fixed throughout"
    assert row(top, "learning_rate")["status"] == "Varied"


def test_missing_values_are_excluded_from_denominator():
    data = frame()
    data.loc[0, P + "learning_rate"] = None
    overall, top, _ = parameter_summaries(data, k=3)
    assert row(overall, "learning_rate", "0.03")["n_runs"] == 2
    result = row(top, "learning_rate")
    assert result["n_runs"] == 2 and result["missing_runs"] == 1
    assert result["mode_share_pct"] == 100
    assert result["mode_share_overall_pct"] == 40


def test_literal_none_is_not_missing():
    data = frame()
    data.loc[0, P + "device"] = "None"
    overall, _, _ = parameter_summaries(data)
    assert row(overall, "device", "None")["n_runs"] == 1


def test_unrecorded_snapshot_still_ranks_but_does_not_supply_values():
    data = frame()
    data.loc[0, "params_recorded"] = False
    _, top, meta = parameter_summaries(data, k=3)
    assert meta["selected_runs"] == 3
    assert row(top, "learning_rate")["missing_runs"] == 1


def test_all_missing_parameter():
    data = frame()
    data[P + "unknown"] = None
    overall, top, _ = parameter_summaries(data)
    assert "model.params.unknown" not in set(overall["parameter"])
    result = row(top, "unknown")
    assert result["mode_values"] == [] and result["n_runs"] == 0
    assert result["status"] == "No recorded values"


def test_tied_scores_break_by_run_id_independent_of_input_order():
    data = frame()
    data["primary_value"] = 1.0
    _, a, _ = parameter_summaries(data, k=1)
    _, b, _ = parameter_summaries(data.iloc[::-1], k=1)
    assert row(a, "learning_rate")["mode_values"] == ["0.03"]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "column", ["silver_dataset_version_id", "split_id", "primary_metric", "experiment_run_id"]
)
def test_mixed_scope_rejected(column):
    data = frame()
    data.loc[0, column] = "other"
    with pytest.raises(ValueError):
        parameter_summaries(data)


def test_missing_cohort_rejected():
    data = frame()
    data.loc[0, "split_id"] = None
    with pytest.raises(ValueError):
        parameter_summaries(data)


def test_duplicate_run_id_rejected():
    data = frame()
    data.loc[1, "run_id"] = "r1"
    with pytest.raises(ValueError):
        parameter_summaries(data)


@pytest.mark.parametrize("k", [0, -1, 2.5, True, "10", None])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        parameter_summaries(frame(), k=k)


def test_numeric_value_order_for_ties():
    data = frame()
    data[P + "depth"] = ["10", "2", "1", "10", "2", "1"]
    _, top, _ = parameter_summaries(data)
    assert row(top, "depth")["mode_values"] == ["1", "2", "10"]


def test_input_not_mutated():
    data = frame()
    before = data.copy(deep=True)
    parameter_summaries(data)
    pd.testing.assert_frame_equal(data, before)


def test_non_default_and_duplicate_index():
    data = frame()
    data.index = [9] * len(data)
    overall, _, _ = parameter_summaries(data)
    assert row(overall, "learning_rate", "0.03")["n_runs"] == 3


def test_no_parameter_columns():
    data = frame().drop(columns=[c for c in frame() if c.startswith("param::")])
    overall, top, meta = parameter_summaries(data)
    assert overall.empty and top.empty and meta["eligible_runs"] == 6


def test_mixed_model_types_reported():
    data = frame()
    data["model_type"] = ["xgb", "lgbm"] * 3
    assert parameter_summaries(data)[2]["model_types"] == ["lgbm", "xgb"]


def test_numeric_score_strings():
    data = frame()
    data["primary_value"] = data["primary_value"].astype(str)
    assert row(parameter_summaries(data)[0], "learning_rate", "0.03")["mean"] == pytest.approx(0.8)
