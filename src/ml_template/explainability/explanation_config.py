"""Explicit local identities and a pinned inference worker; no latest-run selection."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class ExplanationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: StrictInt = 1
    task: Literal["binary_classification", "regression"] = "binary_classification"
    target_unit: StrictStr | None = None
    prediction_atol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    prediction_rtol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    release_dir: StrictStr
    scoring_dir: StrictStr
    expected_release_id: StrictStr
    expected_scoring_batch_id: StrictStr
    runtime_python: StrictStr
    runtime_source_root: StrictStr
    registry_path: StrictStr = "data/project.db"
    output_root: StrictStr = "artifacts/explanations"
    chunk_size: StrictInt = Field(default=256, ge=1)
    max_rows: StrictInt = Field(default=100000, ge=1)
    timeout_seconds: StrictInt = Field(default=600, ge=1)
    batch_definition: StrictStr | None = None
    groups: dict[str, list[str]] = Field(default_factory=dict)
    margin_atol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    margin_rtol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)
    probability_atol: float = Field(default=1e-6, ge=0, allow_inf_nan=False)
    probability_rtol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)

    @field_validator("schema_version")
    @classmethod
    def supported_version(cls, value):
        if value != 1:
            raise ValueError("unsupported explanation schema")
        return value

    @field_validator("expected_release_id", "expected_scoring_batch_id")
    @classmethod
    def explicit_identity(cls, value):
        if not value or value != value.strip() or "/" in value or "\\" in value:
            raise ValueError("explicit nonempty identity required")
        if any(token in value.upper() for token in ("REPLACE", "ACTUAL", "PLACEHOLDER")):
            raise ValueError("replace example identities with actual saved IDs")
        return value

    @field_validator(
        "release_dir",
        "scoring_dir",
        "runtime_python",
        "runtime_source_root",
        "registry_path",
        "output_root",
    )
    @classmethod
    def nonempty_path(cls, value):
        if not value or value != value.strip():
            raise ValueError("nonempty explicit path required")
        return value

    def resolved(self, project_root):
        root = Path(project_root).resolve()
        values = self.model_dump()
        for name in (
            "release_dir",
            "scoring_dir",
            "runtime_python",
            "runtime_source_root",
            "registry_path",
            "output_root",
        ):
            path = Path(values[name]).expanduser()
            absolute = path if path.is_absolute() else root / path
            values[name] = (
                os.path.abspath(absolute) if name == "runtime_python" else str(absolute.resolve())
            )
        result = type(self).model_validate(values)
        output = Path(result.output_root)
        for name in ("release_dir", "scoring_dir", "runtime_source_root"):
            source = Path(getattr(result, name))
            if output == source or output.is_relative_to(source):
                raise ValueError(
                    "output_root cannot be inside a source release, batch or pinned code"
                )
        registry = Path(result.registry_path)
        for name in ("release_dir", "scoring_dir", "runtime_source_root"):
            source = Path(getattr(result, name))
            if registry == source or registry.is_relative_to(source):
                raise ValueError(
                    "registry_path cannot be inside a source release, batch or pinned code"
                )
        return result


def load_config(path, *, project_root):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        out = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in out:
                raise ValueError(f"duplicate YAML key: {key}")
            out[key] = loader.construct_object(value_node, deep=deep)
        return out

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    raw = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    return ExplanationConfig.model_validate(raw).resolved(project_root)
