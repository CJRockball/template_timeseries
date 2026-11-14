"""Strict, dataset-independent source contracts."""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def utc(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ):
        raise ValueError("UTC timestamps must use YYYY-MM-DDTHH:MM:SS[.ffffff]Z")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD")
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError("Noncanonical date")
    return value


@dataclass(frozen=True)
class SourceContract:
    columns: dict[str, str]
    key: tuple[str, ...]
    nullable: tuple[str, ...] = ()
    temporal: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec):
        unknown = set(spec) - {"columns", "key", "nullable", "temporal"}
        if unknown:
            raise ValueError(f"Unknown contract fields: {sorted(unknown)}")
        obj = cls(
            dict(spec["columns"]),
            tuple(spec["key"]),
            tuple(spec.get("nullable", [])),
            dict(spec.get("temporal", {})),
        )
        supported = {"string", "int", "float", "date", "utc_timestamp"}
        if not obj.columns or not obj.key or len(set(obj.key)) != len(obj.key):
            raise ValueError("Nonempty columns and unique key columns required")
        if not set(obj.columns.values()) <= supported:
            raise ValueError("Unsupported dtype")
        if not set(obj.key) <= set(obj.columns) or not set(obj.nullable) <= set(obj.columns):
            raise ValueError("Unknown key/nullable column")
        if set(obj.key) & set(obj.nullable):
            raise ValueError("Keys cannot be nullable")
        if obj.temporal:
            t = obj.temporal
            if set(t) - {"series", "time", "frequency", "gaps", "target"}:
                raise ValueError("Unknown temporal setting")
            series = t.get("series", [])
            time = t.get("time")
            if not series or not set(series) <= set(obj.columns):
                raise ValueError("Temporal series columns required")
            if obj.columns.get(time) != "date":
                raise ValueError("Initial temporal ingestion supports date axes only")
            if t.get("frequency") != "D" or t.get("gaps", "error") not in {"error", "allow"}:
                raise ValueError("Use frequency D and gaps error/allow")
            if set([*series, time]) & set(obj.nullable):
                raise ValueError("Temporal keys cannot be nullable")
            if t.get("target") and obj.columns.get(t["target"]) not in {"int", "float"}:
                raise ValueError("Target must have a numeric declaration")
        return obj

    def to_dict(self):
        return {
            "columns": self.columns,
            "key": list(self.key),
            "nullable": list(self.nullable),
            "temporal": self.temporal,
        }

    def validate(self, frame):
        if set(frame.columns) != set(self.columns) or frame.columns.duplicated().any():
            raise ValueError("Input columns must exactly match contract")
        if frame.empty:
            raise ValueError("Empty source")
        out = frame[sorted(self.columns)].copy()
        for col, kind in self.columns.items():
            vals = []
            for value in out[col].tolist():
                if pd.isna(value):
                    if col not in self.nullable:
                        raise ValueError(f"Null in required column {col}")
                    vals.append(None)
                    continue
                value = str(value)
                if kind == "date":
                    value = iso_date(value)
                elif kind == "utc_timestamp":
                    value = utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
                elif kind == "int":
                    if not re.fullmatch(r"[+-]?\d+", value):
                        raise ValueError(f"Invalid integer in {col}")
                    value = int(value)
                    if not -(2**63) <= value < 2**63:
                        raise ValueError("Integer outside SQLite signed 64-bit range")
                elif kind == "float":
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite value in {col}")
                vals.append(value)
            out[col] = pd.Series(vals, index=out.index, dtype=object)
        if out.duplicated(list(self.key)).any():
            raise ValueError("Duplicate source key")
        if self.temporal:
            t = self.temporal
            keys = [*t["series"], t["time"]]
            if out.duplicated(keys).any():
                raise ValueError("Duplicate series/date observation")
            for _, group in out.groupby(t["series"], dropna=False):
                dates = sorted(date.fromisoformat(x) for x in group[t["time"]])
                if t.get("gaps", "error") == "error" and any(
                    (b - a).days != 1 for a, b in zip(dates, dates[1:], strict=False)
                ):
                    raise ValueError("Undeclared missing daily period")
            if t.get("target") and out[t["target"]].isna().any():
                raise ValueError("Observed-history targets cannot be null")
            order = [*t["series"], t["time"]]
        else:
            order = list(self.key)
        return out.sort_values(order).reset_index(drop=True)
