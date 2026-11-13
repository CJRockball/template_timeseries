"""Opt-in real artifact test. Replays existing predictions; never executes evaluation."""

import os
from pathlib import Path

import pytest


@pytest.mark.local_data
def test_real_release_and_scoring(tmp_path):
    config_path = os.environ.get("ML_TEMPLATE_RELEASE_CONFIG")
    if not config_path:
        pytest.skip("set ML_TEMPLATE_RELEASE_CONFIG to a reviewed release YAML")
    pytest.importorskip("pyarrow")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost")
    from ml_template.db.connection import get_engine
    from ml_template.scoring.batch import score_batch
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.release_builder import build_release, prepare_release
    from ml_template.scoring.release_config import load_release_config

    engine = get_engine()
    request = load_release_config(config_path)
    assert build_release(engine, request, check_only=True)["holdout_labels_loaded"] is False
    material = prepare_release(engine, request)
    result = build_release(engine, request, output_root=tmp_path / "releases")
    assert result["parity"]["shuffled_chunk_parity"]
    folder = Path(result["artifact_dir"])
    predictor = ReleasedPredictor(folder)
    source = material["source"].iloc[:20].reset_index()
    input_path = tmp_path / "new_batch.parquet"
    source.to_parquet(input_path, index=False)
    out = score_batch(
        engine,
        folder,
        input_path,
        output_root=tmp_path / "scores",
        chunk_size=3,
        batch_definition=predictor.policy.batch_definition,
    )
    assert out["n_rows"] == len(source)
    moved = tmp_path / "moved_release"
    folder.rename(moved)
    assert len(ReleasedPredictor(moved).predict(source)) == len(source)
