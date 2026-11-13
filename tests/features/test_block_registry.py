import importlib
import sys

import pandas as pd
import pytest

from ml_template.features import block_registry as reg


@pytest.fixture
def temp_blocks():
    created = []

    def add(name, fn, **kw):
        kw.setdefault("requires", ("a",))
        kw.setdefault("description", "test block")
        reg.register_block(name, **kw)(fn)
        created.append(name)
        return reg.get_block(name)

    yield add
    for name in created:
        reg.unregister_block(name)


def _frame():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.Index([10, 11, 12], name="k"))


def test_builtin_blocks_are_discovered():
    names = reg.discover_blocks()
    assert {"thermal_power", "strain", "rule_margins", "wear_window"} <= set(names)


def test_duplicate_and_bad_names_rejected(temp_blocks):
    temp_blocks("t_dup", lambda df, p: pd.DataFrame({"x": df["a"]}))
    with pytest.raises(ValueError, match="already registered"):
        reg.register_block("t_dup", requires=("a",), description="d")(lambda df, p: df)
    with pytest.raises(ValueError, match="snake case"):
        reg.register_block("Bad-Name", requires=("a",), description="d")


def test_unknown_block_lists_available():
    with pytest.raises(KeyError, match="available"):
        reg.get_block("does_not_exist")


def test_run_enforces_output_contract(temp_blocks):
    f = _frame()
    ok = temp_blocks("t_ok", lambda df, p: pd.DataFrame({"x": df["a"] * 2}))
    assert ok.run(f)["x"].tolist() == [2.0, 4.0, 6.0]

    bad_index = temp_blocks("t_idx", lambda df, p: pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(ValueError, match="same index"):
        bad_index.run(f)

    clash = temp_blocks("t_clash", lambda df, p: pd.DataFrame({"a": df["a"]}))
    with pytest.raises(ValueError, match="duplicates input"):
        clash.run(f)

    nan = temp_blocks("t_nan", lambda df, p: pd.DataFrame({"x": df["a"] / 0.0 * 0.0}))
    with pytest.raises(ValueError, match="NaN or infinite"):
        nan.run(f)

    needs = temp_blocks(
        "t_needs",
        lambda df, p: df[["a"]].rename(columns={"a": "x"}),
        requires=("a", "missing"),
    )
    with pytest.raises(KeyError, match="missing"):
        needs.run(f)


def test_block_id_is_deterministic_and_sensitive(temp_blocks):
    block = temp_blocks(
        "t_id", lambda df, p: pd.DataFrame({"x": df["a"] * p["k"]}), params={"k": 2}
    )
    a = block.block_id("silver_1")
    assert a == block.block_id("silver_1")
    assert a != block.block_id("silver_2")
    assert a != block.block_id("silver_1", {"k": 3})
    assert a.startswith("t_id_")


def test_params_must_be_json_serialisable(temp_blocks):
    block = temp_blocks("t_json", lambda df, p: pd.DataFrame({"x": df["a"]}))
    with pytest.raises(ValueError, match="JSON"):
        block.resolve_params({"bad": object()})


def _write_pkg(tmp_path, name, block_name, extra=None):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for fname, text in (extra or {}).items():
        (pkg / fname).write_text(text)
    (pkg / "blk.py").write_text(
        "import pandas as pd\n"
        "from ml_template.features.block_registry import register_block\n"
        f"@register_block('{block_name}', requires=('a',), description='x')\n"
        "def f(df, params):\n"
        "    return pd.DataFrame({'a2': df['a'] * 2}, index=df.index)\n"
    )
    return pkg


def _cleanup(prefix, block_name):
    reg.unregister_block(block_name)
    for mod in [m for m in sys.modules if m.startswith(prefix)]:
        del sys.modules[mod]


def test_new_block_file_is_found_without_editing_anything(tmp_path, monkeypatch):
    _write_pkg(tmp_path, "scratch_blocks", "t_double_a")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert "t_double_a" in reg.discover_blocks("scratch_blocks")
        assert reg.get_block("t_double_a").run(_frame())["a2"].tolist() == [2.0, 4.0, 6.0]
    finally:
        _cleanup("scratch_blocks", "t_double_a")


def test_helper_module_change_changes_code_hash(tmp_path, monkeypatch):
    pkg = _write_pkg(tmp_path, "hash_blocks", "t_hash", {"_shared.py": "K = 1\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        reg.discover_blocks("hash_blocks")
        first = reg.get_block("t_hash").code_sha256()
        (pkg / "_shared.py").write_text("K = 2\n")
        sys.modules.pop("hash_blocks._shared", None)
        importlib.invalidate_caches()
        assert reg.get_block("t_hash").code_sha256() != first
    finally:
        _cleanup("hash_blocks", "t_hash")
