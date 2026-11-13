"""Guard compatibility without weakening real-engine task checks."""

import pandas as pd
import pytest

from ml_template.dashboard import regression_multi_data as data


def binary():
    return pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "status": ["succeeded"] * 2,
            "primary_value": [0.8, 0.9],
            "primary_metric": ["auc"] * 2,
        }
    )


def test_validated_injected_frame_needs_no_db(monkeypatch):
    monkeypatch.setattr(
        data.base,
        "rows",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected DB query")),
    )
    assert data.nonbinary_launch(None, "launch", frame=binary()) is False


@pytest.mark.parametrize("metrics", [["rmse", "rmse"], ["auc", "rmse"], ["auc", "brier"]])
def test_regression_or_mixed_metrics_blocked(metrics):
    frame = binary()
    frame["primary_metric"] = metrics
    assert data.nonbinary_launch(None, "launch", frame=frame)


def test_explicit_nonbinary_task_blocks_even_binary_metric():
    frame = binary().assign(task="regression")
    assert data.nonbinary_launch(None, "launch", frame=frame)


def test_real_registry_guard_remains_active(monkeypatch):
    engine = object()
    calls = []

    def rows(actual, sql, args):
        assert actual is engine and args == ("launch",)
        calls.append(sql)
        return [{"task": "regression"}]

    monkeypatch.setattr(data.base, "rows", rows)
    monkeypatch.setattr(data.base, "task_hint", lambda row: row["task"])
    assert data.nonbinary_launch(engine, "launch", frame=binary())
    assert calls


def test_missing_or_unknown_semantics_fail_closed():
    with pytest.raises(ValueError):
        data.nonbinary_launch(None, "launch")
    with pytest.raises(ValueError):
        data.nonbinary_launch(None, "launch", frame=binary().assign(primary_metric="unknown_loss"))


def test_empty_unscored_frame_is_safe():
    assert not data.nonbinary_launch(None, "launch", frame=binary().iloc[:0])
