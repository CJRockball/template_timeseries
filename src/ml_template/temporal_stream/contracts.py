"Typed per-row validation with explicit temporal semantics."

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def quote(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid SQL identifier")
    return '"' + value.replace('"', '""') + '"'


def valid_date(value):
    if not re.fullmatch("\\d{4}-\\d{2}-\\d{2}", value):
        raise ValueError("Expected YYYY-MM-DD")
    date.fromisoformat(value)
    return value


def valid_utc(value):
    if not re.fullmatch("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d{1,6})?Z", value):
        raise ValueError("Expected explicit UTC Z timestamp with seconds")
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class SourceContract:
    columns: dict
    key: tuple
    nullable: tuple = ()
    temporal: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if set(value) - {"columns", "key", "nullable", "temporal"}:
            raise ValueError("Unknown contract field")
        obj = cls(
            dict(sorted(value["columns"].items())),
            tuple(value["key"]),
            tuple(value.get("nullable", [])),
            dict(value.get("temporal", {})),
        )
        if not obj.columns or not obj.key or len(set(obj.key)) != len(obj.key):
            raise ValueError("Nonempty unique key required")
        for name in obj.columns:
            quote(name)
            if name.casefold().startswith("__ts_") or name.casefold() in {
                "rowid",
                "oid",
                "_rowid_",
            }:
                raise ValueError("Reserved internal column prefix")
        if len({x.casefold() for x in obj.columns}) != len(obj.columns):
            raise ValueError("SQLite column names must differ case-insensitively")
        if not set(obj.key) <= obj.columns.keys() or not set(obj.nullable) <= obj.columns.keys():
            raise ValueError("Unknown key/nullable column")
        if set(obj.key) & set(obj.nullable):
            raise ValueError("Nullable keys forbidden")
        if not set(obj.columns.values()) <= {"string", "int", "float", "date", "utc_timestamp"}:
            raise ValueError("Unsupported type")
        if obj.temporal:
            t = obj.temporal
            if set(t) - {"series", "time", "frequency", "gaps", "target"}:
                raise ValueError("Unknown temporal setting")
            series = t.get("series", [])
            if not isinstance(series, list) or not series or len(set(series)) != len(series):
                raise ValueError("Temporal series list required")
            time = t.get("time")
            if (
                not set(series) <= obj.columns.keys()
                or time in series
                or obj.columns.get(time) != "date"
            ):
                raise ValueError("Distinct series/date columns required")
            if set([*series, time]) & set(obj.nullable):
                raise ValueError("Temporal keys cannot be nullable")
            if t.get("frequency") != "D" or t.get("gaps", "error") not in {"error", "allow"}:
                raise ValueError("Initial version supports D frequency and error/allow gaps")
            if t.get("target"):
                target = t["target"]
                if obj.columns.get(target) not in {"int", "float"} or target in obj.nullable:
                    raise ValueError("History target must be non-null numeric")
        return obj

    def to_dict(self):
        return {
            "columns": self.columns,
            "key": list(self.key),
            "nullable": list(self.nullable),
            "temporal": self.temporal,
        }

    @property
    def order(self):
        if self.temporal:
            return [*self.temporal["series"], self.temporal["time"]]
        return list(self.key)

    def parse(self, raw):
        result = []
        for name, kind in self.columns.items():
            value = raw[name]
            if value == "":
                if name not in self.nullable:
                    raise ValueError(f"Null in required field {name}")
                result.append(None)
                continue
            if kind == "date":
                value = valid_date(value)
            elif kind == "utc_timestamp":
                value = valid_utc(value)
            elif kind == "int":
                if not re.fullmatch("[+-]?\\d+", value):
                    raise ValueError(f"Invalid integer: {name}")
                value = int(value)
                if not -(2**63) <= value < 2**63:
                    raise ValueError("Integer outside signed 64-bit range")
            elif kind == "float":
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite float")
                if value == 0:
                    value = 0.0
            result.append(value)
        return tuple(result)
