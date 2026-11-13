import pytest

from ml_template.scoring.release_artifacts import inside, read_json, seal, verify, write_json
from ml_template.scoring.release_config import ReleaseConfig


def package(root):
    for name in [
        "release.json",
        "config.json",
        "contract.json",
        "feature_recipe.json",
        "runtime.json",
        "evidence.json",
    ]:
        write_json(root / name, {})
    (root / "model.ubj").write_bytes(b"toy-not-a-model")
    return seal(root)


def test_manifest_identity_and_roundtrip(tmp_path):
    manifest = package(tmp_path)
    assert verify(tmp_path, manifest["release_id"]) == manifest
    assert seal(tmp_path) == manifest


@pytest.mark.parametrize("operation", ["modify", "delete", "add", "identity"])
def test_corruption(tmp_path, operation):
    package(tmp_path)
    if operation == "modify":
        (tmp_path / "model.ubj").write_bytes(b"modified")
    elif operation == "delete":
        (tmp_path / "contract.json").unlink()
    elif operation == "add":
        (tmp_path / "unlisted.txt").write_text("extra")
    else:
        value = read_json(tmp_path / "manifest.json")
        value["release_id"] = "wrong"
        write_json(tmp_path / "manifest.json", value)
    with pytest.raises(ValueError):
        verify(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", ""])
def test_paths(tmp_path, name):
    with pytest.raises(ValueError):
        inside(tmp_path, name)


def test_unexpected_release_id(tmp_path):
    package(tmp_path)
    with pytest.raises(ValueError):
        verify(tmp_path, "rel_other")


def test_duplicate_json_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)


def test_symlink(tmp_path):
    (tmp_path / "target").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        seal(tmp_path)


def test_nested_manifest_is_not_ignored(tmp_path):
    package(tmp_path)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_move(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    expected = package(root)
    root.rename(tmp_path / "moved")
    assert verify(tmp_path / "moved") == expected


def test_acknowledgement_required():
    with pytest.raises(ValueError, match="acknowledgement"):
        ReleaseConfig(
            name="test", evaluation_id="eval", parent_run_id="run", intended_use="research"
        )


def test_no_policy_override():
    with pytest.raises(ValueError):
        ReleaseConfig(
            name="test",
            evaluation_id="eval",
            parent_run_id="run",
            intended_use="research",
            acknowledge_research_release=True,
            threshold=0.5,
        )
