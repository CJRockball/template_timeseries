import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("pyarrow")
pytest.importorskip("dash")
pytest.importorskip("dash_ag_grid")

from sqlalchemy import create_engine  # noqa: E402

from ml_template.dashboard import run_data as rd  # noqa: E402
from ml_template.dashboard.threshold_panels import (  # noqa: E402
    render_single,
    render_threshold_panel,
)
from ml_template.db.tracking_tables import ensure_tracking_tables, splits  # noqa: E402


@pytest.fixture
def data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'threshold.db'}")
    ensure_tracking_tables(engine)
    silver = tmp_path / "silver" / "toy" / "s"
    silver.mkdir(parents=True)
    pd.DataFrame({"key": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 1]}).to_parquet(
        silver / "data.parquet", index=False
    )
    split = tmp_path / "split"
    split.mkdir()
    pd.DataFrame(
        {"key": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"], "fold": [0, 0, 1, 1, -1]}
    ).to_parquet(split / "assignments.parquet", index=False)
    with engine.begin() as conn:
        conn.execute(
            splits.insert(),
            {
                "split_id": "sp",
                "silver_dataset_version_id": "s",
                "path": str(split),
                "policy_json": "{}",
                "key_col": "key",
                "target_col": "target",
            },
        )
    folder = tmp_path / "run"
    folder.mkdir()
    pd.DataFrame(
        {"key": [4, 3, 2, 1], "fold": [1, 1, 0, 0], "p_oof": [0.8, 0.1, 0.9, 0.2]}
    ).to_parquet(folder / "oof_predictions.parquet", index=False)
    header = {"run_id": "r", "split_id": "sp", "silver_dataset_version_id": "s"}
    yield engine, header, folder, tmp_path / "silver", split
    engine.dispose()


def test_complete_dev_join_and_shared_rendering(data):
    engine, header, folder, silver, _ = data
    frame = rd.load_oof_with_labels(engine, header, folder, silver)
    assert frame["key"].tolist() == [1, 2, 3, 4]
    assert frame["y_true"].tolist() == [0, 1, 0, 1]
    before = frame.copy(deep=True)
    assert render_threshold_panel({"r": frame}, 0.5) is not None
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "change", ["missing_row", "missing_label", "duplicate", "wrong_fold", "holdout", "nan_score"]
)
def test_invalid_prediction_population_rejected(data, change):
    engine, header, folder, silver, _ = data
    path = folder / "oof_predictions.parquet"
    table = pd.read_parquet(path)
    if change == "missing_row":
        table = table.iloc[:3]
    elif change == "missing_label":
        table.loc[0, "key"] = 999
    elif change == "duplicate":
        table.loc[0, "key"] = table.loc[1, "key"]
    elif change == "wrong_fold":
        table.loc[0, "fold"] = 0
    elif change == "holdout":
        table.loc[0, "key"] = 5
    else:
        table.loc[0, "p_oof"] = np.nan
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        rd.load_oof_with_labels(engine, header, folder, silver)


def test_missing_split_disables_estimates(data):
    engine, header, folder, silver, split = data
    (split / "assignments.parquet").unlink()
    assert rd.load_oof_with_labels(engine, header, folder, silver) is None


def test_fractional_target_not_silently_cast_to_zero(data):
    engine, header, _, silver, _ = data
    path = silver / "toy" / "s" / "data.parquet"
    pd.DataFrame({"key": [1, 2, 3, 4, 5], "target": [0, 1, 0.3, 1, 1]}).to_parquet(
        path, index=False
    )
    with pytest.raises(ValueError, match="binary"):
        rd.load_labels(engine, header, silver)


def test_empty_and_invalid_single_panels(data, monkeypatch):
    assert render_threshold_panel({}) is not None
    assert render_single(data[0], None, 0.5) is not None
    monkeypatch.setattr(rd, "run_header", lambda *args: data[1])
    monkeypatch.setattr(rd, "run_dir", lambda *args: data[2])

    def invalid(*args):
        raise ValueError("bad OOF coverage")

    monkeypatch.setattr(rd, "load_oof_with_labels", invalid)
    component = render_single(data[0], "r", 0.5)
    assert "bad OOF coverage" in component.children
