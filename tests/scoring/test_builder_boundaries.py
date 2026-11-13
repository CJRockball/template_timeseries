import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

from ml_template.scoring import release_builder as builder


def test_source_projection_never_loads_target(tmp_path, monkeypatch):
    factory = ModuleType("ml_template.models.model_factory")
    factory.prepare_features = lambda frame: frame
    store = ModuleType("ml_template.features.feature_set_store")
    store.resolve_feature_set = lambda *args: (_ for _ in ()).throw(
        AssertionError("unexpected feature store")
    )
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    monkeypatch.setitem(sys.modules, store.__name__, store)
    source = pd.DataFrame({"id": [2, 1], "v": [0.3, 0.2]})
    monkeypatch.setattr(builder, "read_json", lambda path: {"roles": {"feature": ["v"]}})

    def read(path, *, columns):
        assert columns == ["id", "v"]
        return source.copy()

    monkeypatch.setattr(pd, "read_parquet", read)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            dataset_name="test", silver_dataset_version_id="silver", feature_set_id=None
        )
    )
    prepared = SimpleNamespace(run_cfg=cfg, roots={"silver_root": tmp_path}, key_col="id")
    raw, selected = builder._source_frames(prepared)
    assert raw.index.tolist() == [1, 2]
    pd.testing.assert_frame_equal(raw, selected)


def test_parity_reads_scores_not_targets(monkeypatch):
    reference = pd.DataFrame({"id": [1, 2, 3], "p_raw": [0.2, 0.4, 0.6]})

    def read(path, *, columns):
        assert columns == ["id", "p_raw"]
        return reference.copy()

    monkeypatch.setattr(pd, "read_parquet", read)

    class Predictor:
        def predict(self, frame, **kwargs):
            return pd.DataFrame(
                {
                    "id": frame.id.to_numpy(),
                    "p_raw": frame.v.to_numpy(),
                    "p_release": frame.v.to_numpy(),
                }
            )

    source = pd.DataFrame({"v": [0.2, 0.4, 0.6]}, index=pd.Index([1, 2, 3], name="id"))
    prepared = SimpleNamespace(key_col="id", hold_keys=[1, 2, 3])
    spec = SimpleNamespace(
        calibration_id=None, policy=SimpleNamespace(kind="none"), release_variant="raw"
    )
    report = builder._check_parity(
        Predictor(),
        {"prepared": prepared, "spec": spec, "source": source, "evaluation_dir": Path("not-read")},
        1e-7,
    )
    assert report["labels_loaded"] is False
    assert report["max_abs_diff"] == {"p_raw": 0.0}
