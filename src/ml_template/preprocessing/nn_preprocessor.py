"""Fold-fitted, target-free NN preprocessing with portable JSON persistence.

Missing/unknown category IDs are 0/1; observed training categories start at 2.
Every source feature always has a separate 0/1 missing flag. Never refit at
inference. Feature generation runs before this final replacement-capable stage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_template.preprocessing.contracts import validate_estimator_frame, validate_nn_arrays
from ml_template.preprocessing.nn_config import NNPreprocessorConfig

STATE_VERSION = 1


def _category_key(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        parts = ["bool", "true" if value else "false"]
    elif isinstance(value, (int, np.integer)):
        parts = ["number", str(int(value))]
    elif isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("nonfinite categorical value")
        parts = ["number", str(int(value)) if value.is_integer() else value.hex()]
    elif isinstance(value, str):
        parts = ["string", value]
    else:
        raise ValueError(f"unsupported categorical value type: {type(value).__name__}")
    return json.dumps(parts, ensure_ascii=True, separators=(",", ":"))


def _check_key(token: Any) -> None:
    if not isinstance(token, str):
        raise ValueError("invalid saved category key")
    try:
        parts = json.loads(token)
        if not isinstance(parts, list) or len(parts) != 2 or not isinstance(parts[1], str):
            raise ValueError("invalid saved category key")
        kind, text = parts
        if kind == "string":
            value = text
        elif kind == "bool" and text in ("true", "false"):
            value = text == "true"
        elif kind == "number":
            value = float.fromhex(text) if "0x" in text.lower() else int(text)
        else:
            raise ValueError("invalid saved category key")
        if _category_key(value) != token:
            raise ValueError("noncanonical saved category key")
    except (TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError("invalid saved category key") from exc


def _numeric_values(series: pd.Series) -> np.ndarray:
    if series.notna().any() and (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError(f"numeric feature {series.name!r} requires real numeric values")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(values).any():
        raise ValueError(f"infinite numeric input: {series.name}")
    return values


@dataclass(frozen=True)
class NNInputs:
    numeric: np.ndarray
    categorical: np.ndarray
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    cardinalities: tuple[int, ...]


class NNPreprocessor:
    """Fit on an inner-fit slice, outer training fold or final development set.

    The caller controls the training partition. This class never loads targets,
    split assignments, validation data or test data while fitting.
    """

    def __init__(self, config: NNPreprocessorConfig | dict[str, Any]):
        self.config = (
            config
            if isinstance(config, NNPreprocessorConfig)
            else NNPreprocessorConfig.from_dict(config)
        )
        self._state: dict[str, Any] | None = None

    @property
    def missing_names(self) -> tuple[str, ...]:
        return tuple(f"{c}__is_missing" for c in self.config.input_columns)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self.config.input_columns + self.missing_names

    @property
    def numeric_names(self) -> tuple[str, ...]:
        return self.config.numeric + self.missing_names

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return self.config.categorical

    def _require_fitted(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("NNPreprocessor is not fitted")
        return self._state

    @property
    def cardinalities(self) -> tuple[int, ...]:
        vocab = self._require_fitted()["vocabularies"]
        return tuple(len(vocab[c]) + 2 for c in self.config.categorical)

    def _check_frame(self, X: pd.DataFrame, *, fitting: bool = False) -> None:
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("input must be a DataFrame with unique columns")
        if set(X.columns) != set(self.config.input_columns):
            missing = set(self.config.input_columns) - set(X.columns)
            extra = set(X.columns) - set(self.config.input_columns)
            raise ValueError(f"input schema mismatch: missing={missing}, extra={extra}")
        if fitting and len(X) == 0:
            raise ValueError("cannot fit on an empty frame")

    def fit(self, X: pd.DataFrame) -> NNPreprocessor:
        self._state = None
        self._check_frame(X, fitting=True)
        stats = {}
        for column in self.config.numeric:
            values = _numeric_values(X[column])
            observed = values[~np.isnan(values)]
            if len(observed):
                with np.errstate(over="ignore", invalid="ignore"):
                    mean = float(np.mean(observed))
                    std = float(np.std(observed, ddof=0))
                    if self.config.numeric_strategy == "mean":
                        fill = mean
                    elif self.config.numeric_strategy == "median":
                        fill = float(np.median(observed))
                    else:
                        fill = self.config.constant_fill
                center = mean if self.config.scaling == "standard" else 0.0
                scale = std if self.config.scaling == "standard" and std > 0 else 1.0
                if not all(math.isfinite(v) for v in (mean, std, fill, center, scale)):
                    raise ValueError(f"nonfinite fitted numeric statistics: {column}")
            else:
                fill, center, scale = self.config.all_missing_fill, 0.0, 1.0
            stats[column] = {
                "fill": fill,
                "center": center,
                "scale": scale,
                "n_observed": int(len(observed)),
            }
        vocabularies = {
            column: sorted({_category_key(v) for v in X[column].dropna()})
            for column in self.config.categorical
        }
        self._state = {
            "schema_version": STATE_VERSION,
            "config": self.config.to_dict(),
            "numeric_state": stats,
            "vocabularies": vocabularies,
            "fit_rows": len(X),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        state = self._require_fitted()
        self._check_frame(X)
        result: dict[str, Any] = {}
        for column in self.config.numeric:
            stats = state["numeric_state"][column]
            values = _numeric_values(X[column])
            values[np.isnan(values)] = stats["fill"]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                values = ((values - stats["center"]) / stats["scale"]).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite numeric output after float32 conversion: {column}")
            result[column] = values
        for column in self.config.categorical:
            mapping = {key: i + 2 for i, key in enumerate(state["vocabularies"][column])}
            codes = np.empty(len(X), dtype=np.int64)
            missing = X[column].isna().to_numpy()
            for i, value in enumerate(X[column]):
                if missing[i]:
                    codes[i] = 0
                else:
                    code = mapping.get(_category_key(value), 1)
                    if code == 1 and self.config.unknown_policy == "error":
                        raise ValueError(f"unknown category in {column}")
                    codes[i] = code
            result[column] = codes
        for column, flag in zip(self.config.input_columns, self.missing_names, strict=True):
            result[flag] = X[column].isna().to_numpy(dtype=np.float32)
        frame = pd.DataFrame(result, index=X.index)
        if len(frame):
            validate_estimator_frame(frame)
        return frame

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def transform_arrays(self, X: pd.DataFrame) -> NNInputs:
        frame = self.transform(X)
        numeric = np.ascontiguousarray(frame.loc[:, list(self.numeric_names)].to_numpy(np.float32))
        categorical = np.ascontiguousarray(
            frame.loc[:, list(self.categorical_names)].to_numpy(np.int64)
        )
        validate_nn_arrays(numeric, categorical, self.cardinalities)
        return NNInputs(
            numeric, categorical, self.numeric_names, self.categorical_names, self.cardinalities
        )

    def state_dict(self) -> dict[str, Any]:
        """Return an independent JSON-safe checkpoint component."""
        return json.loads(json.dumps(self._require_fitted(), allow_nan=False))

    @classmethod
    def from_state_dict(cls, raw: dict[str, Any]) -> NNPreprocessor:
        try:
            state = json.loads(json.dumps(raw, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise ValueError("preprocessor state must be finite JSON data") from exc
        expected = {"schema_version", "config", "numeric_state", "vocabularies", "fit_rows"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("invalid preprocessor state fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != STATE_VERSION:
            raise ValueError("unsupported preprocessor state version")
        obj = cls(NNPreprocessorConfig.from_dict(state["config"]))
        n = state["fit_rows"]
        if type(n) is not int or n < 1:
            raise ValueError("invalid fit_rows")
        stats = state["numeric_state"]
        vocab = state["vocabularies"]
        if not isinstance(stats, dict) or set(stats) != set(obj.config.numeric):
            raise ValueError("saved numeric state differs from configuration")
        if not isinstance(vocab, dict) or set(vocab) != set(obj.config.categorical):
            raise ValueError("saved vocabulary state differs from configuration")
        for values in stats.values():
            if not isinstance(values, dict) or set(values) != {
                "fill",
                "center",
                "scale",
                "n_observed",
            }:
                raise ValueError("invalid numeric state fields")
            for key in ("fill", "center", "scale"):
                v = values[key]
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError("invalid numeric state value")
            count = values["n_observed"]
            if type(count) is not int or not 0 <= count <= n or values["scale"] <= 0:
                raise ValueError("invalid observed count or scale")
            if obj.config.scaling == "none" and (values["center"] != 0 or values["scale"] != 1):
                raise ValueError("invalid unscaled numeric state")
        for tokens in vocab.values():
            if not isinstance(tokens, list):
                raise ValueError("invalid saved vocabulary")
            for token in tokens:
                _check_key(token)
            if tokens != sorted(set(tokens)):
                raise ValueError("saved vocabulary must have unique canonical sorted keys")
        obj._state = state
        return obj

    def save(self, path: str | Path) -> None:
        """Write a new standalone JSON artifact. Never overwrite an existing file."""
        state = self.state_dict()
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> NNPreprocessor:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        with Path(path).open(encoding="utf-8") as handle:
            state = json.load(handle, object_pairs_hook=unique_object)
        return cls.from_state_dict(state)
