import math

import pandas as pd
import pytest

from ml_template.features.block_registry import discover_blocks, get_block
from ml_template.features.blocks import _ai4i_common as c

discover_blocks()


def _raw():
    """Four hand-computed rows indexed by the entity key."""
    return pd.DataFrame(
        {
            "type": pd.Categorical(["L", "M", "H", "L"]),
            "air_temperature_k": [300.0, 300.0, 299.0, 300.0],
            "process_temperature_k": [310.0, 308.0, 308.0, 310.0],
            "rotational_speed_rpm": [1500.0, 1300.0, 2800.0, 1400.0],
            "torque_nm": [40.0, 50.0, 60.0, 10.0],
            "tool_wear_min": [100.0, 220.0, 250.0, 10.0],
        },
        index=pd.Index([1, 2, 3, 4], name="udi"),
    )


POWER = [2000 * math.pi, 65000 * math.pi / 30, 5600 * math.pi, 14000 * math.pi / 30]


def test_thermal_power_values():
    out = get_block("thermal_power").run(_raw())
    assert list(out.columns) == ["temp_diff_k", "power_w"]
    assert out["temp_diff_k"].tolist() == pytest.approx([10.0, 8.0, 9.0, 10.0])
    assert out["power_w"].tolist() == pytest.approx(POWER)  # torque x rpm x 2 pi / 60


def test_thermal_power_sanity_counts_hdf_rows():
    block = get_block("thermal_power")
    raw = _raw()
    out = block.run(raw)
    # only row 2 has diff 8.0 K < 8.6 K and 1300 rpm < 1380 rpm
    assert block.sanity(raw, out, {}) == {"hdf_rule_rows": 1.0}


def test_strain_values_and_type_limits():
    out = get_block("strain").run(_raw())
    assert out["wear_torque_minnm"].tolist() == pytest.approx([4000, 11000, 15000, 100])
    # limits: L 11000, M 12000, H 13000
    assert out["osf_margin_minnm"].tolist() == pytest.approx([7000, 1000, -2000, 10900])


def test_strain_unknown_type_rejected():
    raw = _raw().assign(type=["L", "M", "X", "L"])
    with pytest.raises(ValueError, match="X"):
        get_block("strain").run(raw)


def test_strain_limits_are_parameters():
    out = get_block("strain").run(_raw(), {"osf_limit_minnm": {"L": 1.0, "M": 2.0, "H": 3.0}})
    expected = [1 - 4000, 2 - 11000, 3 - 15000, 1 - 100]
    assert out["osf_margin_minnm"].tolist() == pytest.approx(expected)


def test_rule_margins_values():
    out = get_block("rule_margins").run(_raw())
    assert out["hdf_temp_margin_k"].tolist() == pytest.approx([1.4, -0.6, 0.4, 1.4])
    assert out["hdf_speed_margin_rpm"].tolist() == pytest.approx([120, -80, 1420, 20])
    assert out["pwf_low_margin_w"].tolist() == pytest.approx([x - 3500 for x in POWER])
    assert out["pwf_high_margin_w"].tolist() == pytest.approx([9000 - x for x in POWER])


def test_rule_margins_sanity_flags_each_rule():
    block = get_block("rule_margins")
    raw = _raw()
    observed = block.sanity(raw, block.run(raw), {})
    # HDF: row 2. PWF: row 3 (above 9000 W) and row 4 (below 3500 W).
    assert observed == {"hdf_rule_rows": 1.0, "pwf_rule_rows": 2.0}


def test_wear_window_values():
    out = get_block("wear_window").run(_raw())
    assert out["twf_wear_margin_min"].tolist() == pytest.approx([-100, 20, 50, -190])
    assert out["twf_wear_in_window"].tolist() == [0, 1, 1, 0]


@pytest.mark.parametrize("name", ["thermal_power", "strain", "rule_margins", "wear_window"])
def test_blocks_are_pure_and_use_feature_columns_only(name):
    raw = _raw()
    before = raw.copy()
    block = get_block(name)
    out = block.run(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert out.index.equals(raw.index)
    assert set(block.requires) <= set(c.FEATURES)


def test_blocks_run_without_any_outcome_or_target_column():
    raw = _raw()
    assert "machine_failure" not in raw.columns
    for name in ("thermal_power", "strain", "rule_margins", "wear_window"):
        get_block(name).run(raw)
