import pandas as pd
import pytest

from ml_template.features.group_statistics import GroupStatisticsStep


def _train():
    return pd.DataFrame(
        {"type": ["L", "L", "M", "M", "M"], "v": [1.0, 3.0, 10.0, 20.0, 30.0]},
        index=pd.Index([1, 2, 3, 4, 5], name="udi"),
    )


def _step(**kw):
    kw.setdefault("group_by", ["type"])
    kw.setdefault("columns", ["v"])
    return GroupStatisticsStep(**kw)


def test_statistics_come_from_training_rows_only():
    step = _step(stats=["mean", "std"], deviation=True).fit(_train())
    val = pd.DataFrame({"type": ["L", "M"], "v": [10.0, 5.0]}, index=pd.Index([9, 10], name="udi"))
    out = step.transform(val)
    assert out["v__by_type__mean"].tolist() == [2.0, 20.0]
    assert out["v__by_type__std"].tolist() == pytest.approx([2**0.5, 10.0])
    assert out["v__by_type__dev"].tolist() == [8.0, -15.0]
    assert list(out.columns) == step.output_columns()
    assert out.index.equals(val.index)


def test_unseen_and_small_groups_fall_back_to_overall_statistics():
    step = _step(stats=["mean"], min_group_size=3, count=True, frequency=True).fit(_train())
    val = pd.DataFrame({"type": ["L", "M", "H"], "v": [0.0, 0.0, 0.0]})
    out = step.transform(val)
    overall = _train()["v"].mean()
    assert out["v__by_type__mean"].tolist() == [overall, 20.0, overall]
    assert out["grp_type__count"].tolist() == [2, 3, 0]
    assert out["grp_type__freq"].tolist() == pytest.approx([0.4, 0.6, 0.0])


def test_singleton_std_and_zscore_are_defined():
    train = pd.DataFrame({"type": ["L", "M", "M"], "v": [5.0, 1.0, 3.0]})
    step = _step(stats=["std"], zscore=True).fit(train)
    out = step.transform(train)
    assert out["v__by_type__std"].notna().all()
    assert out["v__by_type__z"].notna().all()


def test_two_group_columns_and_input_not_modified():
    train = _train().assign(shift=["a", "b", "a", "b", "a"])
    before = train.copy()
    step = _step(group_by=["type", "shift"], stats=["mean"]).fit(train)
    out = step.transform(train)
    assert "v__by_type_shift__mean" in out.columns
    pd.testing.assert_frame_equal(train, before)


def test_bad_arguments_rejected():
    with pytest.raises(ValueError, match="unknown stats"):
        _step(stats=["mode"])
    with pytest.raises(ValueError, match="min_group_size"):
        _step(min_group_size=0)
    with pytest.raises(KeyError, match="ghost"):
        _step(columns=["ghost"]).fit(_train())
