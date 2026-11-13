"""Actual temporary Git worktrees and isolated stdlib venvs; uv sync is stubbed."""

import json
import platform
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_template.scoring import release_artifacts
from ml_template.scripts import prepare_runtime as helper


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    root = repo / "src/ml_template/scoring"
    root.mkdir(parents=True)
    (repo / "src/ml_template/__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    (root / "release_artifacts.py").write_bytes(Path(release_artifacts.__file__).read_bytes())
    (repo / "pyproject.toml").write_text("[project]\nname='runtime-fixture'\nversion='0.1.0'\n")
    (repo / "uv.lock").write_text("# fixture; actual uv installation is stubbed\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "src", "pyproject.toml", "uv.lock"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture baseline"], check=True)
    commit = helper.resolve_commit(repo, "HEAD")
    runtime = {
        "python": platform.python_version(),
        "packages": {name: "not-installed" for name in release_artifacts.PACKAGES},
        "code_sha256": helper.checkout_hashes(repo),
    }
    release = tmp_path / "release"
    release.mkdir()
    for name in (
        "release.json",
        "config.json",
        "contract.json",
        "feature_recipe.json",
        "evidence.json",
    ):
        (release / name).write_text("{}")
    (release / "runtime.json").write_text(json.dumps(runtime))
    (release / "model.ubj").write_bytes(b"fixture model bytes; never deserialized")
    release_artifacts.seal(release)
    calls = []

    def fake_sync(destination, python_version, timeout):
        calls.append((destination, python_version))
        venv.EnvBuilder(with_pip=False).create(destination / ".venv")
        executable = destination / ".venv/bin/python"
        site = subprocess.check_output(
            [str(executable), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
        Path(site, "fixture_editable.pth").write_text(str(destination / "src") + "\n")

    monkeypatch.setattr(helper, "sync_environment", fake_sync)
    return SimpleNamespace(
        repo=repo,
        release=release,
        runtime=runtime,
        commit=commit,
        root=tmp_path / "runtimes",
        calls=calls,
    )


def test_commit_hashes_match_saved_source(runtime_fixture):
    obj = runtime_fixture
    assert helper.commit_hashes(obj.repo, obj.commit) == obj.runtime["code_sha256"]
    assert helper.matching_commit(obj.repo, obj.runtime["code_sha256"]) == obj.commit


def test_check_is_read_only(runtime_fixture):
    obj = runtime_fixture
    result = helper.prepare(obj.repo, obj.release, runtime_root=obj.root)
    assert result["ready"] is False
    assert not obj.root.exists()
    assert not obj.calls


def test_create_then_reuse_real_isolated_venv(runtime_fixture):
    obj = runtime_fixture
    result = helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    assert result["ready"] and result["created"]
    assert Path(result["preparation_manifest"]).is_file()
    assert Path(result["yaml_values"]["runtime_python"]).is_file()
    assert len(obj.calls) == 1
    second = helper.prepare(obj.repo, obj.release, runtime_root=obj.root)
    assert second["ready"] and not second["created"]
    assert len(obj.calls) == 1
    assert helper.verify_environment(Path(result["checkout"]), obj.release)["ready"]


def test_existing_checkout_can_be_adopted(runtime_fixture):
    obj = runtime_fixture
    first = helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    result = helper.prepare(
        obj.repo, obj.release, runtime_root=obj.root, checkout=first["checkout"], create=True
    )
    assert result["ready"] and not result["created"]
    assert len(obj.calls) == 1


def test_changed_source_is_not_overwritten(runtime_fixture):
    obj = runtime_fixture
    result = helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    target = Path(result["checkout"]) / "src/ml_template/new.py"
    target.write_text("# additional source\n")
    with pytest.raises(ValueError, match="different source"):
        helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    assert target.read_text() == "# additional source\n"


def test_incomplete_existing_venv_is_not_repaired(runtime_fixture):
    obj = runtime_fixture
    result = helper.prepare(obj.repo, obj.release, runtime_root=obj.root)
    destination = Path(result["checkout"])
    destination.parent.mkdir()
    helper.git(obj.repo, "worktree", "add", "--detach", str(destination), obj.commit)
    (destination / ".venv").mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    assert not obj.calls


def test_wrong_commit_rejected_without_writes(runtime_fixture):
    obj = runtime_fixture
    (obj.repo / "src/ml_template/change.py").write_text("# changed\n")
    helper.git(obj.repo, "add", "src")
    helper.git(obj.repo, "commit", "-qm", "different source")
    with pytest.raises(ValueError, match="No exact"):
        helper.prepare(obj.repo, obj.release, runtime_root=obj.root, commit="HEAD", create=True)
    assert not obj.root.exists()
    assert helper.matching_commit(obj.repo, obj.runtime["code_sha256"]) == obj.commit


def test_destination_inside_dev_rejected(runtime_fixture):
    obj = runtime_fixture
    with pytest.raises(ValueError, match="outside"):
        helper.prepare(obj.repo, obj.release, checkout=obj.repo / "pinned")


def test_old_interpreter_rejects_modified_source(runtime_fixture):
    obj = runtime_fixture
    result = helper.prepare(obj.repo, obj.release, runtime_root=obj.root, create=True)
    (Path(result["checkout"]) / "src/ml_template/__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="does not match"):
        helper.verify_environment(result["checkout"], obj.release)
