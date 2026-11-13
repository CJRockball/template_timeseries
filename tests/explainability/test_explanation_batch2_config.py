import sys

import pytest

from ml_template.explainability.explanation_config import ExplanationConfig, load_config


def values():
    return {
        "release_dir": "release",
        "scoring_dir": "scores",
        "expected_release_id": "rel_fixture",
        "expected_scoring_batch_id": "score_fixture",
        "runtime_python": sys.executable,
        "runtime_source_root": "pinned",
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("chunk_size", 0),
        ("chunk_size", True),
        ("chunk_size", 1.5),
        ("max_rows", 0),
        ("timeout_seconds", 0),
        ("schema_version", 2),
        ("expected_release_id", "rel_REPLACE_ME"),
        ("expected_release_id", "../rel_bad"),
        ("expected_scoring_batch_id", ""),
        ("margin_atol", -1),
        ("probability_atol", float("nan")),
        ("extra_unknown", True),
    ],
)
def test_config_rejects_invalid(name, value):
    raw = values()
    raw[name] = value
    with pytest.raises(ValueError):
        ExplanationConfig.model_validate(raw)


def test_python_symlink_preserved(tmp_path):
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    raw = values()
    raw["runtime_python"] = str(executable)
    config = ExplanationConfig(**raw).resolved(tmp_path)
    assert config.runtime_python == str(executable)
    assert config.runtime_python != str(executable.resolve())


@pytest.mark.parametrize("source", ["release_dir", "scoring_dir", "runtime_source_root"])
def test_output_cannot_modify_inputs(tmp_path, source):
    raw = values()
    raw["output_root"] = raw[source] + "/outputs"
    with pytest.raises(ValueError, match="output_root"):
        ExplanationConfig(**raw).resolved(tmp_path)


def test_duplicate_yaml_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("chunk_size: 1\nchunk_size: 2\n")
    with pytest.raises(ValueError, match="duplicate YAML"):
        load_config(path, project_root=tmp_path)


def test_registry_cannot_modify_release(tmp_path):
    raw = values()
    raw["registry_path"] = "release/project.db"
    with pytest.raises(ValueError, match="registry_path"):
        ExplanationConfig(**raw).resolved(tmp_path)
