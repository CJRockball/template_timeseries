import pandas as pd
import pytest

from ml_template.features.block_attribution import attribute_importance

SOURCES = {"torque_nm": "silver", "air_temperature_k": "silver", "power_w": "thermal_power"}


def test_groups_by_source_and_sorts():
    imp = pd.DataFrame(
        {"feature": ["torque_nm", "air_temperature_k", "power_w"], "value": [0.1, 0.2, 0.5]}
    )
    out = attribute_importance(imp, SOURCES)
    assert out["source"].tolist() == ["thermal_power", "silver"]
    assert out["sum_value"].tolist() == pytest.approx([0.5, 0.3])
    assert out["n_features"].tolist() == [1, 2]
    assert out["max_value"].tolist() == pytest.approx([0.5, 0.2])


def test_unknown_feature_rejected():
    imp = pd.DataFrame({"feature": ["ghost"], "value": [1.0]})
    with pytest.raises(ValueError, match="ghost"):
        attribute_importance(imp, SOURCES)
