import math

import pandas as pd
import pytest

from ml_template.features.block_registry import discover_blocks, get_block

discover_blocks()


def _df():
    return pd.DataFrame(
        {
            "type": pd.Categorical(["L", "M", "H", "L"]),
            "shift": ["a", "a", "b", "b"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        },
        index=pd.Index([1, 2, 3, 4], name="udi"),
    )


def _run(**params):
    return get_block("generic_interactions").run(_df(), params)


def test_unary_transforms():
    out = _run(transforms=[{"columns": ["x"], "ops": ["square", "cube", "log1p", "signed_log"]}])
    assert out["x__square"].tolist() == [1, 4, 9, 16]
    assert out["x__cube"].tolist() == [1, 8, 27, 64]
    assert out["x__log1p"].tolist() == pytest.approx([math.log1p(v) for v in [1, 2, 3, 4]])
    assert out["x__signed_log"].tolist() == pytest.approx([math.log1p(v) for v in [1, 2, 3, 4]])


def test_domain_violations_raise():
    df = _df().assign(x=[1.0, 0.0, -1.0, 2.0])
    block = get_block("generic_interactions")
    for op in ("log", "sqrt", "reciprocal"):
        with pytest.raises(ValueError, match="column 'x'"):
            block.run(df, {"transforms": [{"columns": ["x"], "ops": [op]}]})
    with pytest.raises(ValueError, match="denominator"):
        block.run(df, {"numeric_pairs": [{"a": "y", "b": "x", "ops": ["ratio"]}]})


def test_numeric_pairs():
    out = _run(numeric_pairs=[{"a": "x", "b": "y", "ops": ["product", "ratio", "difference"]}])
    assert out["x__x__y"].tolist() == [2, 8, 18, 32]
    assert out["x__div__y"].tolist() == pytest.approx([0.5] * 4)
    assert out["x__minus__y"].tolist() == [-1, -2, -3, -4]


def test_numeric_category_gating_uses_fixed_levels():
    out = _run(numeric_category=[{"numeric": "x", "category": "type", "levels": ["L", "M"]}])
    assert list(out.columns) == ["x__if_type_L", "x__if_type_M"]
    assert out["x__if_type_L"].tolist() == [1, 0, 0, 4]
    assert out["x__if_type_M"].tolist() == [0, 2, 0, 0]


def test_category_pairs_make_combined_labels():
    out = _run(category_pairs=[{"a": "type", "b": "shift"}])
    assert out["type__x__shift"].tolist() == ["L|a", "M|a", "H|b", "L|b"]


def test_required_columns_are_recorded_and_checked():
    block = get_block("generic_interactions")
    params = {"numeric_pairs": [{"a": "x", "b": "y", "ops": ["product"]}]}
    assert block.required_columns(params) == ("x", "y")
    with pytest.raises(KeyError, match="ghost"):
        block.run(_df(), {"transforms": [{"columns": ["ghost"], "ops": ["square"]}]})


def test_bad_configuration_is_rejected():
    block = get_block("generic_interactions")
    with pytest.raises(ValueError, match="nothing to build"):
        block.run(_df())
    with pytest.raises(ValueError, match="unknown transform ops"):
        block.run(_df(), {"transforms": [{"columns": ["x"], "ops": ["cuberoot"]}]})
    with pytest.raises(ValueError, match="must be numeric"):
        block.run(_df(), {"transforms": [{"columns": ["shift"], "ops": ["square"]}]})
    dup = {"transforms": [{"columns": ["x", "x"], "ops": ["square"]}]}
    with pytest.raises(ValueError, match="duplicate output"):
        block.run(_df(), dup)
