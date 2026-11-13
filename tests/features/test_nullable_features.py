"""Nullable stateless FE; no fitted preprocessing or external data required."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml_template.features.block_registry import Block, get_block
from ml_template.features.blocks import generic_interactions  # noqa: F401
from ml_template.features.feature_recipe import replay_recipe


def frame():
    return pd.DataFrame(
        {
            "x": pd.array([2, None, 8, 16], dtype="Int64"),
            "y": pd.array([1, 2, None, 4], dtype="Int64"),
            "cat": pd.array(["A", "B", None, "A"], dtype="string"),
            "other": pd.Categorical(["C", None, "D", "C"]),
        },
        index=pd.Index([21, 3, 18, 7], name="id"),
    )


def block(fn, nullable=()):
    return Block("test_nullable", fn, (), "test", nullable_outputs=nullable)


def params():
    return {
        "numeric_pairs": [{"a": "x", "b": "y", "ops": ["difference"]}],
        "nullable_outputs": ["x__minus__y"],
    }


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "Int32", "boolean", "string"])
def test_explicit_nullable_outputs_preserve_dtype_and_mask(dtype):
    f = frame()
    values = ["ok", None, "yes", "no"] if dtype == "string" else [1, None, 0, 1]
    expected = pd.DataFrame({"out": pd.array(values, dtype=dtype)}, index=f.index)
    result = block(lambda df, p: expected.copy(), ["out"]).run(f)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "string", "category"])
def test_strict_by_default(dtype):
    values = ["ok", None, "yes", "no"] if dtype in {"string", "category"} else [1, None, 0, 1]
    f = frame()
    b = block(lambda df, p: pd.DataFrame({"out": pd.array(values, dtype=dtype)}, index=df.index))
    with pytest.raises(ValueError, match="non-nullable outputs"):
        b.run(f)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("nullable", [(), ("out",)])
def test_infinity_is_never_allowed(value, nullable):
    b = block(lambda df, p: pd.DataFrame({"out": [value] * len(df)}, index=df.index), nullable)
    with pytest.raises(ValueError, match="infinite values"):
        b.run(frame())


def test_nullable_permission_does_not_leak_to_other_columns():
    b = block(lambda df, p: pd.DataFrame({"a": np.nan, "b": np.nan}, index=df.index), ["a"])
    with pytest.raises(ValueError, match="non-nullable outputs.*b"):
        b.run(frame())


@pytest.mark.parametrize("policy", [None, True, "out", {"out"}, ["out", "out"], [3], [""]])
def test_bad_policy_rejected(policy):
    with pytest.raises(ValueError, match="nullable_outputs"):
        block(lambda df, p: pd.DataFrame({"out": 1}, index=df.index), policy).run(frame())


def test_typo_rejected_even_when_no_values_are_missing():
    with pytest.raises(ValueError, match="unknown nullable outputs"):
        block(lambda df, p: pd.DataFrame({"out": 1}, index=df.index), ["typo"]).run(frame())


def test_all_missing_and_empty_input():
    b = block(
        lambda df, p: pd.DataFrame(
            {"out": pd.array([None] * len(df), dtype="Int64")}, index=df.index
        ),
        ["out"],
    )
    assert b.run(frame())["out"].isna().all()
    assert b.run(frame().iloc[:0]).empty


@pytest.mark.parametrize("kind", ["index", "collision", "duplicate", "empty", "not_frame"])
def test_other_output_guards_remain(kind):
    f = frame()
    functions = {
        "index": lambda df, p: pd.DataFrame({"out": [1] * len(df)}),
        "collision": lambda df, p: df[["x"]],
        "duplicate": lambda df, p: pd.DataFrame(
            [[1, 2]] * len(df), columns=["z", "z"], index=df.index
        ),
        "empty": lambda df, p: pd.DataFrame(index=df.index),
        "not_frame": lambda df, p: df["x"],
    }
    with pytest.raises(ValueError):
        block(functions[kind]).run(f)


@pytest.mark.parametrize(
    "op", ["square", "cube", "sqrt", "log", "log1p", "signed_log", "reciprocal"]
)
def test_unary_transforms_propagate_input_nulls(op):
    f = frame()
    name = f"x__{op}"
    result = get_block("generic_interactions").run(
        f,
        {
            "transforms": [{"columns": ["x"], "ops": [op]}],
            "nullable_outputs": [name],
        },
    )
    assert result[name].isna().equals(f["x"].isna())
    assert np.isfinite(result[name].dropna()).all()


@pytest.mark.parametrize("op,tag", [("difference", "minus"), ("product", "x"), ("ratio", "div")])
def test_numeric_pairs_propagate_either_missing_input(op, tag):
    f = frame()
    name = f"x__{tag}__y"
    result = get_block("generic_interactions").run(
        f,
        {
            "numeric_pairs": [{"a": "x", "b": "y", "ops": [op]}],
            "nullable_outputs": [name],
        },
    )
    assert result[name].isna().equals(f["x"].isna() | f["y"].isna())


@pytest.mark.parametrize("op,bad", [("log", 0), ("sqrt", -1), ("log1p", -1), ("reciprocal", 0)])
def test_domain_violations_still_raise_alongside_nulls(op, bad):
    f = frame()
    f.loc[21, "x"] = bad
    with pytest.raises(ValueError, match="column 'x'"):
        get_block("generic_interactions").run(
            f,
            {
                "transforms": [{"columns": ["x"], "ops": [op]}],
                "nullable_outputs": [f"x__{op}"],
            },
        )


def test_zero_denominator_is_not_relabelled_as_missing():
    f = frame()
    f.loc[21, "y"] = 0
    with pytest.raises(ValueError, match="denominator"):
        get_block("generic_interactions").run(
            f,
            {
                "numeric_pairs": [{"a": "x", "b": "y", "ops": ["ratio"]}],
                "nullable_outputs": ["x__div__y"],
            },
        )


@pytest.mark.parametrize("op", ["reciprocal", "signed_log", "square"])
def test_infinite_numeric_inputs_cannot_be_hidden_by_transforms(op):
    f = frame().astype({"x": "float64"})
    f.loc[21, "x"] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        get_block("generic_interactions").run(
            f,
            {
                "transforms": [{"columns": ["x"], "ops": [op]}],
                "nullable_outputs": [f"x__{op}"],
            },
        )


def test_category_cross_keeps_real_nulls_and_literal_nan_label():
    f = frame()
    f.loc[7, "cat"] = "nan"
    result = get_block("generic_interactions").run(
        f,
        {
            "category_pairs": [{"a": "cat", "b": "other"}],
            "nullable_outputs": ["cat__x__other"],
        },
    )
    assert result.iloc[0, 0] == "A|C"
    assert result.iloc[3, 0] == "nan|C"
    assert result.iloc[1:3, 0].isna().all()


def test_numeric_category_gate_preserves_missingness_from_either_input():
    f = frame()
    result = get_block("generic_interactions").run(
        f,
        {
            "numeric_category": [{"numeric": "x", "category": "cat", "levels": ["A", "B"]}],
            "nullable_outputs": ["x__if_cat_A", "x__if_cat_B"],
        },
    )
    assert result.iloc[0].tolist() == [2, 0]
    assert result.iloc[1:3].isna().all().all()
    assert result.iloc[3].tolist() == [16, 0]


def test_generic_stays_strict_without_opt_in():
    with pytest.raises(ValueError, match="non-nullable outputs"):
        get_block("generic_interactions").run(
            frame(),
            {
                "numeric_pairs": [{"a": "x", "b": "y", "ops": ["difference"]}],
            },
        )


def test_nullable_policy_participates_in_identity():
    b = get_block("generic_interactions")
    strict = {"numeric_pairs": params()["numeric_pairs"]}
    assert b.block_id("silver", params()) == b.block_id("silver", params())
    assert b.block_id("silver", strict) != b.block_id("silver", params())
    base = block(lambda df, p: pd.DataFrame({"z": 1}, index=df.index))
    nullable = block(base.fn, ["z"])
    assert base.block_id("silver") != nullable.block_id("silver")


def test_registry_implementation_participates_in_hash(monkeypatch):
    from ml_template.features import block_registry as reg

    b = get_block("generic_interactions")
    before = b.code_sha256()
    read = reg._source
    monkeypatch.setattr(
        reg, "_source", lambda mod: read(mod) + ("\n# changed validator" if mod is reg else "")
    )
    assert b.code_sha256() != before


def frozen_recipe(f):
    b = get_block("generic_interactions")
    p = b.resolve_params(params())
    return json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "raw_columns": list(f.columns),
                "columns": [*f.columns, "x__minus__y"],
                "blocks": [
                    {
                        "block_name": b.name,
                        "code_sha256": b.code_sha256(),
                        "params": p,
                        "derived_from": list(b.required_columns(p)),
                        "columns": ["x__minus__y"],
                    }
                ],
            }
        )
    )


def test_frozen_replay_matches_build_and_does_not_mutate_source():
    f = frame()
    before = f.copy(deep=True)
    expected = pd.concat([f, get_block("generic_interactions").run(f, params())], axis=1)
    assert_frame_equal(replay_recipe(f, frozen_recipe(f)), expected)
    assert_frame_equal(f, before)


def test_chunk_singleton_shuffle_and_new_key_parity():
    f = frame()
    recipe = frozen_recipe(f)
    whole = replay_recipe(f, recipe)
    singletons = pd.concat([replay_recipe(f.iloc[i : i + 1], recipe) for i in range(len(f))])
    assert_frame_equal(singletons, whole)
    shuffled = replay_recipe(f.sample(frac=1, random_state=7), recipe).reindex(f.index)
    assert_frame_equal(shuffled, whole)
    new = f.copy()
    new.index = pd.Index([101, 102, 103, 104], name="id")
    result = replay_recipe(new, recipe)
    assert result.index.equals(new.index)
    assert result["x__minus__y"].isna().equals(whole["x__minus__y"].isna().set_axis(new.index))


def test_replay_rejects_changed_code_hash():
    f = frame()
    recipe = frozen_recipe(f)
    recipe["blocks"][0]["code_sha256"] = "not-current"
    with pytest.raises(ValueError, match="implementation changed"):
        replay_recipe(f, recipe)
