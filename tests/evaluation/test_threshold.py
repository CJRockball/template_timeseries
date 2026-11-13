import numpy as np
import pytest

from ml_template.evaluation.threshold import (
    NONE_THRESHOLD,
    budget_boundaries,
    fold_recommendations,
    operating_point,
    recommendations,
    threshold_profile,
)


@pytest.mark.parametrize("seed", range(8))
def test_exact_profile_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=35)
    p = rng.integers(0, 11, size=35) / 10
    table = threshold_profile(y, p)
    for row in table.to_dict("records"):
        flags = p >= row["threshold"]
        tp, fp = int((flags & (y == 1)).sum()), int((flags & (y == 0)).sum())
        fn, tn = int((~flags & (y == 1)).sum()), int((~flags & (y == 0)).sum())
        assert [row[k] for k in ("tp", "fp", "fn", "tn")] == [tp, fp, fn, tn]
        for beta in (1, 2):
            expected = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)
            assert row[f"f{beta}"] == pytest.approx(expected)


def test_distinct_scores_not_coarse_slider_grid():
    p = np.array([0.1234567, 0.1234568])
    table = threshold_profile([0, 1], p)
    assert len(table) == 5
    selected = recommendations(table).set_index("objective").loc["f1"]
    assert selected["threshold"] == 0.1234568
    assert selected["objective_value"] == 1


def test_highest_candidate_tiebreak():
    table = threshold_profile([0, 1], [0.2, 0.8])
    # 0.8 is the highest candidate yielding the perfect one-positive decision.
    assert recommendations(table).iloc[0]["threshold"] == 0.8


def test_constant_scores_have_one_indivisible_tie():
    table = threshold_profile([0, 1, 0, 1], [0.5] * 4)
    row = recommendations(table).iloc[0]
    assert row["threshold"] == 0.5 and row["flagged"] == 4
    assert row["objective_value"] == pytest.approx(2 / 3)


def test_zero_and_one_scores_and_flag_none():
    table = threshold_profile([0, 1], [0, 1])
    assert table.iloc[-1]["threshold"] == NONE_THRESHOLD
    assert table.iloc[-1]["flagged"] == 0
    assert operating_point([0, 1], [0, 1], 1)["tp"] == 1
    assert operating_point([0, 1], [0, 1], NONE_THRESHOLD)["flagged"] == 0


def test_no_positives_is_undefined_not_spurious_optimum():
    table = threshold_profile([0, 0], [0.2, 0.8])
    assert table["f1"].isna().all()
    assert recommendations(table)["threshold"].isna().all()


def test_all_positives_is_explicit():
    row = recommendations(threshold_profile([1, 1], [0.2, 0.8])).iloc[0]
    assert row["objective_value"] == 1 and row["threshold"] == 0.2
    assert "no negative" in row["status"]


def test_profile_is_permutation_invariant():
    a = threshold_profile([0, 1, 1], [0.4, 0.8, 0.4])
    b = threshold_profile([1, 0, 1], [0.4, 0.4, 0.8])
    assert a.equals(b)


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0], [0, 1]),
        ([2], [0.5]),
        ([0], [float("nan")]),
        ([0], [1.2]),
        ([[0]], [[0.5]]),
        ([0.5], [0.5]),
    ],
)
def test_invalid_inputs_rejected(y, p):
    with pytest.raises(ValueError):
        threshold_profile(y, p)


def test_foldwise_optima_include_undefined_folds():
    table = fold_recommendations([0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert set(table["fold"]) == {0, 1}
    assert table[table["fold"] == 0]["threshold"].isna().all()
    with pytest.raises(ValueError):
        fold_recommendations([0, 1], [0.2, 0.8], [-1, 0])


def test_near_envelopes_use_absolute_score_tolerance():
    table = threshold_profile([0, 1, 0, 1], [0.1, 0.8, 0.6, 0.7])
    result = recommendations(table, tolerance=0.02)
    for row in result.to_dict("records"):
        eligible = table[table[row["objective"]] >= row["objective_value"] - 0.02 - 1e-12]
        assert row["near_candidate_min"] == eligible["threshold"].min()
        assert row["near_candidate_max"] == eligible["threshold"].max()
    with pytest.raises(ValueError):
        recommendations(table, tolerance=-0.1)


def test_budget_boundary_tie_is_not_a_scalar_cutoff():
    result = budget_boundaries([1, 0, 1, 0], [0.8, 0.8, 0.8, 0.1], rates=[0.25]).iloc[0]
    assert result["flagged"] == 1 and result["tp"] == 1
    assert not result["threshold_exact"]
    assert result["flagged_if_score_ge_boundary"] == 3
    assert result["boundary_tie_count"] == 3


def test_exact_budget_and_ceil_rounding():
    p = np.linspace(1, 0, 100)
    y = np.arange(100) % 2
    row = budget_boundaries(y, p, [0.07]).iloc[0]
    assert row["flagged"] == 7 and row["threshold_exact"]
    all_rows = budget_boundaries([0, 1], [0.5, 0.5], [1]).iloc[0]
    assert all_rows["threshold_exact"] and all_rows["flagged"] == 2


def test_invalid_budget_and_threshold():
    with pytest.raises(ValueError):
        budget_boundaries([0, 1], [0.2, 0.8], [0])
    with pytest.raises(ValueError):
        operating_point([0, 1], [0.2, 0.8], -1)
