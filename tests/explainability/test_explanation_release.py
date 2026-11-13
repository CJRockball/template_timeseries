from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ml_template.explainability.release import explain_release
from ml_template.explainability.result import build_margin_explanation


class ToyAdapter:
    def explain_margin(self, X, *, tolerance=None):
        phi = (X * 0.2).to_numpy()
        margin = -1 + phi.sum(axis=1)
        return build_margin_explanation(
            X, phi, np.full(len(X), -1.0), margin, expit(margin), tolerance=tolerance
        )


class ToyPredictor:
    """Service-boundary fake, not evidence of real release/model integration."""

    def __init__(self):
        self.model = ToyAdapter()
        self.release_id = "rel_toy"
        self.manifest = {"content_sha256": "toy"}
        self.info = {
            "parent_run_id": "run_toy",
            "calibration_id": "cal_toy",
            "release_variant": "calibrated",
        }
        self.contract = {"key_col": "entity"}
        self.prediction_calls = 0
        self.preparation_calls = 0

    def prepare_features(self, frame):
        self.preparation_calls += 1
        if frame.entity.duplicated().any():
            raise ValueError("duplicate keys")
        return frame.set_index("entity")[["x"]].copy()

    def predict(self, frame, *, chunk_size, include_policy):
        self.prediction_calls += 1
        if frame.entity.duplicated().any():
            raise ValueError("duplicate keys")
        p = expit(-1 + frame.x.to_numpy() * 0.2)
        calibrated = p * 0.5
        flags = np.zeros(len(frame), dtype=bool)
        flags[np.argmax(p)] = True
        return pd.DataFrame(
            {
                "entity": frame.entity.to_numpy(),
                "p_raw": p,
                "p_calibrated": calibrated,
                "p_release": calibrated,
                "policy_flag": flags,
                "release_id": self.release_id,
            }
        )


def batch():
    return pd.DataFrame({"entity": [30, 10, 20], "x": [1.0, 3.0, 2.0]})


def test_complete_batch_policy_only_once():
    predictor = ToyPredictor()
    out = explain_release(predictor, batch(), chunk_size=1)
    assert predictor.prediction_calls == 1
    assert predictor.preparation_calls == 3
    assert out.predictions.policy_flag.sum() == 1
    assert out.margin.features.index.tolist() == [30, 10, 20]
    assert out.metadata["persisted_scoring_batch_verified"] is False
    assert out.metadata["calibrated_probability_attribution"] is False


def test_chunks_and_order_are_equivalent():
    a = explain_release(ToyPredictor(), batch())
    b = explain_release(ToyPredictor(), batch(), chunk_size=1)
    pd.testing.assert_frame_equal(a.margin.contributions, b.margin.contributions)
    c = explain_release(ToyPredictor(), batch().iloc[::-1], chunk_size=2)
    pd.testing.assert_frame_equal(
        a.margin.contributions.sort_index(), c.margin.contributions.sort_index()
    )


def test_singleton():
    out = explain_release(ToyPredictor(), batch().iloc[:1])
    assert len(out.margin.rows) == 1


@pytest.mark.parametrize("size", [True, 0, -1, 1.5])
def test_invalid_chunk_size(size):
    with pytest.raises(ValueError, match="chunk_size"):
        explain_release(ToyPredictor(), batch(), chunk_size=size)


def test_duplicate_keys_rejected():
    X = batch()
    X.loc[1, "entity"] = 30
    with pytest.raises(ValueError):
        explain_release(ToyPredictor(), X)


def test_empty_batch_rejected():
    with pytest.raises(ValueError):
        explain_release(ToyPredictor(), batch().iloc[:0])


def test_unsupported_adapter():
    predictor = ToyPredictor()
    predictor.model = SimpleNamespace()
    with pytest.raises(TypeError, match="support"):
        explain_release(predictor, batch())


def test_wrong_probabilities_rejected():
    predictor = ToyPredictor()
    original = predictor.predict

    def wrong(*args, **kwargs):
        out = original(*args, **kwargs)
        out["p_raw"] = 0.99
        return out

    predictor.predict = wrong
    with pytest.raises(ValueError, match="probability"):
        explain_release(predictor, batch())


def test_changed_explanation_keys_rejected():
    predictor = ToyPredictor()
    original = predictor.model.explain_margin

    def wrong(X, **kwargs):
        X = X.copy()
        X.index = X.index + 100
        return original(X, **kwargs)

    predictor.model.explain_margin = wrong
    with pytest.raises(ValueError, match="keys"):
        explain_release(predictor, batch())


def test_wrong_release_identity_rejected():
    predictor = ToyPredictor()
    original = predictor.predict

    def wrong(*args, **kwargs):
        out = original(*args, **kwargs)
        out["release_id"] = "other"
        return out

    predictor.predict = wrong
    with pytest.raises(ValueError, match="identity"):
        explain_release(predictor, batch())


def test_dropped_prediction_rejected():
    predictor = ToyPredictor()
    original = predictor.predict
    predictor.predict = lambda *a, **k: original(*a, **k).iloc[:1]
    with pytest.raises(ValueError, match="coverage"):
        explain_release(predictor, batch())
