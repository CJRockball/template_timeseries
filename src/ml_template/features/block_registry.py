"""Registry of feature blocks.

A block is a pure function: it receives the Silver feature columns (indexed by the entity
key) and the block parameters, and returns new columns on the same index. It must not read
files, the target or other rows' statistics, and must not fit anything on data. Anything
fitted (scaling, encoding, selection) belongs inside the training folds.

Blocks register themselves with ``@register_block`` and are found by ``discover_blocks``.
Missing outputs are forbidden unless named by the block's ``nullable_outputs`` policy.
The policy permits nulls only; it never permits infinity or performs imputation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

BLOCKS_PACKAGE = "ml_template.features.blocks"
_NAME = re.compile(r"[a-z][a-z0-9_]*")

BlockFn = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]
SanityFn = Callable[[pd.DataFrame, pd.DataFrame, Mapping[str, Any]], dict[str, float]]
NullableOutputs = Sequence[str] | Callable[[Mapping[str, Any]], Sequence[str]]


def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError(f"block parameters must be JSON-serialisable: {exc}") from exc


def _source(module: ModuleType) -> str:
    """Source text read from the file, so edits are seen without a stale line cache."""
    return Path(module.__file__).read_text(encoding="utf-8")


def _helper_sources(module: ModuleType) -> list[str]:
    """Sources of the ``_*`` helper modules next to a block module."""
    package = getattr(module, "__package__", "") or ""
    pkg = sys.modules.get(package)
    if pkg is None or not hasattr(pkg, "__path__"):
        return []
    sources = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            helper = importlib.import_module(f"{package}.{info.name}")
            sources.append(_source(helper))
    return sources


@dataclass(frozen=True)
class Block:
    name: str
    fn: BlockFn
    requires: (
        tuple[str, ...] | Callable[[Mapping[str, Any]], Sequence[str]]
    )  # input columns the block reads (derived-from)
    description: str
    params: dict[str, Any] = field(default_factory=dict)  # defaults
    sanity: SanityFn | None = None
    expected: dict[str, float] = field(default_factory=dict)  # expected sanity values

    nullable_outputs: NullableOutputs = ()

    def missing_columns(self, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Resolve explicitly nullable output names; no batch-dependent permissions."""
        policy = self.nullable_outputs
        columns = policy(self.resolve_params(params)) if callable(policy) else policy
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise ValueError("nullable_outputs must be a sequence of output column names")
        if any(not isinstance(c, str) or not c for c in columns):
            raise ValueError("nullable_outputs must contain non-empty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("nullable_outputs contains duplicate output names")
        return tuple(columns)

    def required_columns(self, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        merged = self.resolve_params(params)
        cols = self.requires(merged) if callable(self.requires) else self.requires
        return tuple(cols)

    def code_sha256(self) -> str:
        """Hash the block, its helpers and the shared execution/validation implementation."""
        module = sys.modules[self.fn.__module__]
        parts = [_source(module), *_helper_sources(module), _source(sys.modules[__name__])]
        return hashlib.sha256("\n#---\n".join(parts).encode()).hexdigest()

    def resolve_params(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = {**self.params, **(overrides or {})}
        _canonical_json(merged)  # fail early on non-serialisable values
        return merged

    def block_id(self, silver_id: str, params: Mapping[str, Any] | None = None) -> str:
        """Content address: same Silver, same code and same parameters give the same ID."""
        payload = {
            "name": self.name,
            "silver": silver_id,
            "code": self.code_sha256(),
            "params": self.resolve_params(params),
            "nullable_outputs": list(self.missing_columns(params)),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        return f"{self.name}_{digest[:12]}"

    def run(self, features: pd.DataFrame, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Compute the block and check the output contract."""
        missing = [c for c in self.required_columns(params) if c not in features.columns]
        if missing:
            raise KeyError(f"block {self.name!r} needs columns {missing}")
        allowed_missing = set(self.missing_columns(params))
        out = self.fn(features, self.resolve_params(params))
        if not isinstance(out, pd.DataFrame) or out.shape[1] == 0:
            raise ValueError(f"block {self.name!r} must return a non-empty DataFrame")
        if not out.index.equals(features.index):
            raise ValueError(f"block {self.name!r} must return the same index as its input")
        clash = sorted(set(out.columns) & set(features.columns))
        if clash:
            raise ValueError(f"block {self.name!r} duplicates input columns {clash}")
        if out.columns.duplicated().any():
            raise ValueError(f"block {self.name!r} returns duplicate column names")
        unknown = sorted(allowed_missing - set(out.columns))
        if unknown:
            raise ValueError(f"block {self.name!r} declares unknown nullable outputs {unknown}")
        numeric = out.select_dtypes(include="number")
        for column in numeric:
            observed = numeric[column].dropna().to_numpy(dtype=float)
            if not np.isfinite(observed).all():
                raise ValueError(
                    f"block {self.name!r} produced NaN or infinite values in {column!r}"
                )
        unexpected = [c for c in out if out[c].isna().any() and c not in allowed_missing]
        if unexpected:
            reason = (
                "NaN or infinite values"
                if any(c in numeric for c in unexpected)
                else "missing values"
            )
            raise ValueError(
                f"block {self.name!r} produced {reason} in non-nullable outputs {unexpected}"
            )
        return out


_REGISTRY: dict[str, Block] = {}


def register_block(
    name: str,
    *,
    requires: Sequence[str] | Callable[[Mapping[str, Any]], Sequence[str]],
    description: str,
    params: Mapping[str, Any] | None = None,
    sanity: SanityFn | None = None,
    expected: Mapping[str, float] | None = None,
    nullable_outputs: NullableOutputs = (),
) -> Callable[[BlockFn], BlockFn]:
    """Decorator that adds a block function to the registry."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"block name {name!r} must be lower snake case")

    def decorate(fn: BlockFn) -> BlockFn:
        if name in _REGISTRY:
            raise ValueError(f"block {name!r} is already registered")
        _REGISTRY[name] = Block(
            name=name,
            fn=fn,
            requires=requires if callable(requires) else tuple(requires),
            description=description,
            params=dict(params or {}),
            sanity=sanity,
            expected=dict(expected or {}),
            nullable_outputs=nullable_outputs,
        )
        return fn

    return decorate


def unregister_block(name: str) -> None:
    """Remove a block. Meant for tests."""
    _REGISTRY.pop(name, None)


def discover_blocks(package: str = BLOCKS_PACKAGE) -> list[str]:
    """Import every public module of the blocks package so it can register its blocks."""
    pkg = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package}.{info.name}")
    return available_blocks()


def available_blocks() -> list[str]:
    return sorted(_REGISTRY)


def get_block(name: str) -> Block:
    if name not in _REGISTRY:
        discover_blocks()
    if name not in _REGISTRY:
        raise KeyError(f"unknown block {name!r}; available: {available_blocks()}")
    return _REGISTRY[name]
