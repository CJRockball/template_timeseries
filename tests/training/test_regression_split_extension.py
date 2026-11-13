import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_template.training.run_config import SplitConfig
from ml_template.training.split import compute_split_id, holdout_keys, make_split, save_split


def data():
    return pd.DataFrame({"id": np.arange(1030), "y": np.linspace(-2.25, 85.75, 1030)})


def test_coverage_order_and_seed():
    df = data()
    cfg = SplitConfig(method="random")
    a = make_split(df, cfg, "id", "y")
    pd.testing.assert_frame_equal(a, make_split(df.sample(frac=1, random_state=7), cfg, "id", "y"))
    assert len(a) == 1030 and a.id.is_unique
    assert (a.partition == "holdout").sum() == 206
    assert sorted(a.loc[a.partition == "dev", "fold"].unique()) == list(range(5))
    assert (a.loc[a.partition == "holdout", "fold"] == -1).all()
    b = make_split(df, SplitConfig(method="random", cv_seed=99), "id", "y")
    assert set(holdout_keys(a, "id")) == set(holdout_keys(b, "id"))
    assert not a.fold.equals(b.fold)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_bad_target(bad):
    df = data()
    df.loc[0, "y"] = bad
    with pytest.raises(ValueError):
        make_split(df, SplitConfig(method="random"), "id", "y")


def test_binary_unchanged():
    df = pd.DataFrame({"id": np.arange(1000), "y": np.tile([0, 1], 500)})
    cfg = SplitConfig()
    a = make_split(df, cfg, "id", "y")
    y = df.y.to_numpy()
    dev, _ = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
    dev = np.sort(dev)
    expected = np.full(len(df), -1)
    for k, (_, val) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(dev, y[dev])
    ):
        expected[dev[val]] = k
    np.testing.assert_array_equal(a.fold, expected)
    assert compute_split_id("s", cfg, "id", "y") != compute_split_id(
        "s", SplitConfig(method="random"), "id", "y"
    )


def test_manifest_reuse(tmp_path):
    df = data()
    cfg = SplitConfig(method="random")
    a = make_split(df, cfg, "id", "y")
    sid, _, m = save_split(a, "s", cfg, "id", "y", df.y, tmp_path)
    assert m["task"] == "regression" and "n_positive_total" not in m
    assert save_split(a, "s", cfg, "id", "y", df.y, tmp_path)[0] == sid
