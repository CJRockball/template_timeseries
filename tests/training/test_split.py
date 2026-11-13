import numpy as np
import pandas as pd
import pytest

from ml_template.training import split as sp
from ml_template.training.run_config import SplitConfig

KEY, TARGET = "udi", "machine_failure"


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 1000
    return pd.DataFrame({KEY: np.arange(1, n + 1), TARGET: (rng.random(n) < 0.05).astype(int)})


def test_partition_sizes_and_stratification(df):
    a = sp.make_split(df, SplitConfig(), KEY, TARGET)
    assert len(a) == len(df)
    assert (a["partition"] == "holdout").sum() == 200
    assert set(a.loc[a["partition"] == "dev", "fold"]) == set(range(5))
    assert (a.loc[a["partition"] == "holdout", "fold"] == -1).all()
    y = df.set_index(KEY)[TARGET]
    hold_rate = y[a.loc[a["partition"] == "holdout", KEY]].mean()
    assert hold_rate == pytest.approx(y.mean(), abs=0.01)


def test_deterministic_and_order_independent(df):
    cfg = SplitConfig()
    a = sp.make_split(df, cfg, KEY, TARGET)
    b = sp.make_split(df.sample(frac=1, random_state=3), cfg, KEY, TARGET)
    pd.testing.assert_frame_equal(a, b)
    c = sp.make_split(df, SplitConfig(seed=7), KEY, TARGET)
    assert not a["fold"].equals(c["fold"])


def test_folds_are_disjoint_and_exclude_holdout(df):
    a = sp.make_split(df, SplitConfig(), KEY, TARGET)
    hold = set(sp.holdout_keys(a, KEY))
    seen = set()
    for train, val in sp.cv_folds(a, KEY):
        assert not hold & (set(train) | set(val))
        assert not set(train) & set(val)
        assert not seen & set(val)
        seen |= set(val)
    assert seen | hold == set(df[KEY])


def test_split_id_depends_on_all_inputs():
    base = sp.compute_split_id("s1", SplitConfig(), KEY, TARGET)
    assert base == sp.compute_split_id("s1", SplitConfig(), KEY, TARGET)
    assert base != sp.compute_split_id("s2", SplitConfig(), KEY, TARGET)
    assert base != sp.compute_split_id("s1", SplitConfig(n_folds=3), KEY, TARGET)
    assert base != sp.compute_split_id("s1", SplitConfig(), "other", TARGET)


def test_bad_input_rejected(df):
    cfg = SplitConfig()
    with pytest.raises(ValueError, match="not unique"):
        sp.make_split(pd.concat([df, df.iloc[:1]]), cfg, KEY, TARGET)
    bad = df.copy()
    bad.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="binary"):
        sp.make_split(bad, cfg, KEY, TARGET)
    with pytest.raises(KeyError):
        sp.make_split(df, cfg, "missing", TARGET)


def test_too_few_positives_rejected():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({KEY: np.arange(200), TARGET: (rng.random(200) < 0.04).astype(int)})
    with pytest.raises(ValueError, match="positives"):
        sp.make_split(df, SplitConfig(), KEY, TARGET)


def test_save_load_roundtrip_and_immutability(df, tmp_path):
    pytest.importorskip("pyarrow")
    cfg = SplitConfig()
    a = sp.make_split(df, cfg, KEY, TARGET)
    sid, path, manifest = sp.save_split(a, "silver_x", cfg, KEY, TARGET, df[TARGET], tmp_path)
    assert sid == sp.compute_split_id("silver_x", cfg, KEY, TARGET)
    pd.testing.assert_frame_equal(sp.load_split(sid, tmp_path), a)
    sid2, _, _ = sp.save_split(a, "silver_x", cfg, KEY, TARGET, None, tmp_path)
    assert sid2 == sid
    changed = a.copy()
    changed.loc[0, "fold"] = 4 if a.loc[0, "fold"] != 4 else 3
    with pytest.raises(FileExistsError):
        sp.save_split(changed, "silver_x", cfg, KEY, TARGET, None, tmp_path)
    assert manifest["n_holdout"] == 200
    assert manifest["n_positive_holdout"] <= manifest["n_positive_total"]


def test_load_missing_split_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the split step"):
        sp.load_split("split_nope", tmp_path)


def test_register_split_is_idempotent(df, tmp_path):
    pytest.importorskip("pyarrow")
    sqlalchemy = pytest.importorskip("sqlalchemy")
    cfg = SplitConfig()
    a = sp.make_split(df, cfg, KEY, TARGET)
    sid, path, manifest = sp.save_split(a, "silver_x", cfg, KEY, TARGET, df[TARGET], tmp_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 't.db'}")
    sp.register_split(engine, sid, path, manifest)
    sp.register_split(engine, sid, path, manifest)
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text("SELECT split_id, n_holdout FROM splits")).all()
    assert rows == [(sid, 200)]
