import pytest

from ml_template.explainability.artifacts import (
    file_entries,
    read_json,
    safe_path,
    seal_explanation,
    verify_explanation,
    verify_files,
)


@pytest.mark.parametrize("name", ["../bad", "/bad", "a/../b", "a\\b", "a//b", ""])
def test_unsafe_path_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_intermediate_symlink_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        safe_path(root, "link/file")


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_invalid_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)


def test_inventory_rejects_modified_or_extra_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    entries = file_entries(tmp_path)
    verify_files(tmp_path, entries)
    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="unlisted"):
        verify_files(tmp_path, entries)


def test_manifest_roundtrip_and_tamper(tmp_path):
    names = (
        "config.json",
        "summary.json",
        "provenance.json",
        "rows.parquet",
        "features.parquet",
        "contributions.parquet",
        "global_summary.parquet",
        "feature_metadata.json",
        "verification.json",
        "report.md",
        "report.html",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"fixture bytes: inventory test, not Parquet data")
    manifest = seal_explanation(tmp_path, "explain_fixture")
    assert verify_explanation(tmp_path, "explain_fixture") == manifest
    with pytest.raises(ValueError, match="identity"):
        verify_explanation(tmp_path, "different")
    (tmp_path / "rows.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_explanation(tmp_path)
