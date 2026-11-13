import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from ml_template.models.xgb_classifier import XGBAdapter  # noqa: E402
from ml_template.training.run_config import XGBParams  # noqa: E402


def test_learning_curve_shape_and_training_trend():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=300) > 0.5).astype(int))
    adapter = XGBAdapter(XGBParams(), seed=1)
    curve = adapter.learning_curve(
        X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], 20, "logloss"
    )
    assert set(curve["dataset"]) == {"train", "val"}
    assert curve.groupby("dataset")["round"].max().tolist() == [20, 20]
    train = curve[curve["dataset"] == "train"]["value"].to_numpy()
    assert train[-1] < train[0]
