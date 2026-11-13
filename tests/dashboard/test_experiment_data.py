import pandas as pd
import pytest

from ml_template.dashboard import experiment_data as ed


def _data():
    rows, params = [], []
    for i, (name, seed, value, depth) in enumerate(
        [
            ("base", 1, 0.70, 3),
            ("base", 2, 0.80, 3),
            ("base", 1, 0.72, 3),
            ("base", 3, 0.99, 4),
            ("new", 1, 0.85, 3),
            ("new", 2, 0.87, 3),
        ]
    ):
        rid = f"r{i}"
        rows.append(
            {
                "run_id": rid,
                "experiment_name": name,
                "seed": seed,
                "primary_value": value,
                "primary_metric": "pr_auc",
                "status": "succeeded",
                "silver_dataset_version_id": "s1",
                "split_id": "split1",
                "started_at": f"2026-10-15T0{i}:00:00Z",
                "duration_s": 10.0 + i,
                "git_sha": "abc",
                "git_dirty": 0,
            }
        )
        for key, value in {
            "seed": seed,
            "experiment_name": name,
            "model.params.max_depth": depth,
            "split.seed": 42,
            "analysis.learning_curves": bool(i % 2),
        }.items():
            params.append({"run_id": rid, "key": key, "value": str(value)})
    return pd.DataFrame(rows), pd.DataFrame(params)


def test_seed_and_analysis_do_not_split_recipe():
    runs, params = _data()
    prepared = ed.prepare_runs(runs, params)
    assert prepared.loc[[0, 1, 2], "recipe_id"].nunique() == 1
    assert prepared.loc[[0, 3], "recipe_id"].nunique() == 2


def test_leaderboard_and_latest_seed_deduplication():
    runs, params = _data()
    prepared = ed.prepare_runs(runs, params)
    board = ed.seed_leaderboard(prepared)
    base = board[(board.experiment_name == "base") & (board.n_seeds == 2)].iloc[0]
    assert base["mean"] == pytest.approx(0.76)
    assert base["std"] == pytest.approx(pd.Series([0.72, 0.80]).std())
    assert base["excluded_repeats"] == 1
    assert list(board["rank"]) == [1, 2, 3]
    assert board.iloc[0]["mean"] == pytest.approx(0.99)
    assert pd.isna(board.iloc[0]["std"])


def test_lower_is_better():
    runs, params = _data()
    board = ed.seed_leaderboard(ed.prepare_runs(runs, params), higher_is_better=False)
    assert board.iloc[0]["mean"] == pytest.approx(0.76)


def test_failed_and_running_runs_are_not_scored():
    runs, params = _data()
    runs.loc[0, "status"] = "failed"
    runs.loc[1, "status"] = "running"
    prepared = ed.prepare_runs(runs, params)
    board = ed.seed_leaderboard(prepared)
    base = board[(board.experiment_name == "base") & (board["mean"] < 0.9)].iloc[0]
    assert base["n_seeds"] == 1
    inventory = ed.experiment_inventory(prepared).set_index("experiment_name")
    assert inventory.loc["base", "failed"] == 1
    assert inventory.loc["base", "running"] == 1
    assert inventory.loc["base", "latest_sgt"] == "2026-10-15 11:00:00"


def test_mixed_cohorts_cannot_be_ranked_together():
    runs, params = _data()
    runs.loc[0, "split_id"] = "split2"
    prepared = ed.prepare_runs(runs, params)
    with pytest.raises(ValueError, match="Select one"):
        ed.seed_leaderboard(prepared)
    options = ed.cohort_options(prepared)
    assert len(options) == 2
    assert ed.select_cohort(prepared, options[0]["value"])["split_id"].nunique() == 1


def test_missing_params_remain_isolated():
    runs, params = _data()
    prepared = ed.prepare_runs(runs, params.iloc[:0])
    assert prepared.recipe_id.nunique() == len(runs)
    assert not prepared.params_recorded.any()


def test_unseeded_run_does_not_inflate_sample_size():
    runs, params = _data()
    runs.loc[1, "seed"] = None
    board = ed.seed_leaderboard(ed.prepare_runs(runs, params))
    base = board[(board.experiment_name == "base") & (board["mean"] < 0.9)].iloc[0]
    assert base["n_seeds"] == 1
    assert base["n_scored_runs"] == 3
    assert base["excluded_repeats"] == 1


def test_split_seed_is_not_excluded():
    runs, params = _data()
    params.loc[(params.run_id == "r1") & (params.key == "split.seed"), "value"] = "43"
    prepared = ed.prepare_runs(runs, params)
    assert prepared.loc[[0, 1], "recipe_id"].nunique() == 2


def test_empty_database_frames():
    runs, params = _data()
    prepared = ed.prepare_runs(runs.iloc[:0], params.iloc[:0])
    assert ed.experiment_inventory(prepared).empty
    assert ed.seed_leaderboard(prepared).empty
    assert ed.cohort_options(prepared) == []


def test_duplicate_params_fail_loudly():
    runs, params = _data()
    with pytest.raises(ValueError, match="Duplicate"):
        ed.prepare_runs(runs, pd.concat([params, params.iloc[:1]]))


def test_recipe_separates_code_revisions():
    runs, params = _data()
    runs.loc[1, "git_sha"] = "def"
    prepared = ed.prepare_runs(runs, params)
    assert prepared.loc[[0, 1], "recipe_id"].nunique() == 2
