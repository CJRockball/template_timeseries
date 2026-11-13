import pytest

from ml_template.tracking import run_artifacts as ra


def test_create_run_dir_refuses_duplicates(tmp_path):
    ra.create_run_dir("run_a", tmp_path)
    with pytest.raises(FileExistsError):
        ra.create_run_dir("run_a", tmp_path)


def test_manifest_lists_files_and_detects_change(tmp_path):
    run_dir = ra.create_run_dir("run_a", tmp_path)
    (run_dir / "model.ubj").write_bytes(b"model-bytes")
    (run_dir / "sub").mkdir()
    (run_dir / "sub" / "notes.txt").write_text("x")
    manifest = ra.write_artifact_manifest(run_dir)
    assert set(manifest["files"]) == {"model.ubj", "sub/notes.txt"}
    ra.verify_artifact(run_dir, "model.ubj")
    (run_dir / "model.ubj").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        ra.verify_artifact(run_dir, "model.ubj")


def test_manifest_rewrite_excludes_itself(tmp_path):
    run_dir = ra.create_run_dir("run_a", tmp_path)
    (run_dir / "a.txt").write_text("a")
    ra.write_artifact_manifest(run_dir)
    assert set(ra.write_artifact_manifest(run_dir)["files"]) == {"a.txt"}


def test_verify_without_manifest_or_entry(tmp_path):
    run_dir = ra.create_run_dir("run_a", tmp_path)
    (run_dir / "a.txt").write_text("a")
    with pytest.raises(FileNotFoundError):
        ra.verify_artifact(run_dir, "a.txt")
    ra.write_artifact_manifest(run_dir)
    with pytest.raises(KeyError):
        ra.verify_artifact(run_dir, "missing.txt")


def test_environment_info_has_python_and_packages():
    info = ra.environment_info()
    assert info["python"] and "numpy" in info["packages"]


def test_stored_path_round_trip(tmp_path):
    outside = tmp_path / "run_a"
    outside.mkdir()
    stored = ra.relative_to_root(outside)
    assert ra.resolve_stored_path(stored) == outside.resolve()
    inside = ra.ROOT / "artifacts" / "runs" / "run_x"
    assert ra.relative_to_root(inside) == "artifacts/runs/run_x"
    assert ra.resolve_stored_path("artifacts/runs/run_x") == inside
