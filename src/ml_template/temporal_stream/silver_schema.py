"Dependency-light Silver semantics, types and canonical hashing (no Arrow import)."

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime

FORMAT = "temporal-silver-v1"


HASH_FORMAT = "temporal-silver-typed-jsonl-v1"


KINDS = {"string", "int", "float", "date", "utc_timestamp"}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quote(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("Invalid column identifier")
    return '"' + name.replace('"', '""') + '"'


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def checked_date(value):
    if not isinstance(value, str) or not re.fullmatch("\\d{4}-\\d{2}-\\d{2}", value):
        raise ValueError("Date must be YYYY-MM-DD")
    date.fromisoformat(value)
    return value


def checked_utc(value):
    if not isinstance(value, str) or not re.fullmatch(
        "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d{1,6})?Z", value
    ):
        raise ValueError("Timestamp must have explicit UTC Z and seconds")
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class TemporalSchema:
    spec: dict

    @classmethod
    def from_dict(cls, value):
        spec = json.loads(canonical(value))
        allowed = {
            "columns",
            "nullable",
            "observation_id",
            "series",
            "time",
            "target",
            "features",
            "known_future",
            "available_at",
            "frequency",
            "gaps",
            "target_delay_periods",
            "origin_policy",
        }
        required = {
            "columns",
            "observation_id",
            "series",
            "time",
            "target",
            "features",
            "frequency",
            "gaps",
            "target_delay_periods",
            "origin_policy",
        }
        if set(spec) - allowed or not required <= set(spec):
            raise ValueError("Unknown or missing temporal schema field")
        cols = spec["columns"]
        if not isinstance(cols, dict) or not cols or (not set(cols.values()) <= KINDS):
            raise ValueError("Nonempty supported column declarations required")
        for col in cols:
            quote(col)
            if col.casefold().startswith("__ts_"):
                raise ValueError("Reserved column prefix")
        if len({c.casefold() for c in cols}) != len(cols):
            raise ValueError("Case-insensitive column collision")
        for name in ["nullable", "series", "features", "known_future"]:
            vals = spec.setdefault(name, [])
            if (
                not isinstance(vals, list)
                or any(not isinstance(v, str) for v in vals)
                or len(vals) != len(set(vals))
                or (not set(vals) <= cols.keys())
            ):
                raise ValueError(f"Invalid column list: {name}")
        if not spec["series"]:
            raise ValueError("Series columns required")
        role_columns = [spec["observation_id"], *spec["series"], spec["time"], spec["target"]]
        if spec.get("available_at"):
            role_columns.append(spec["available_at"])
        if len(set(role_columns)) != len(role_columns) or not set(role_columns) <= cols.keys():
            raise ValueError("Semantic roles must be distinct declared columns")
        if set(role_columns) & set(spec["nullable"]):
            raise ValueError("Identity/time/target/availability roles cannot be nullable")
        if set(spec["features"]) & set(role_columns):
            raise ValueError(
                "Semantic identity/time/target fields cannot be automatic model features"
            )
        if not set(spec["known_future"]) <= set(spec["features"]):
            raise ValueError("Known-future fields must be declared features")
        if cols[spec["time"]] != "date" or cols[spec["target"]] not in {"int", "float"}:
            raise ValueError("Daily date axis and numeric target required")
        if spec.get("available_at") and cols[spec["available_at"]] != "utc_timestamp":
            raise ValueError("Availability column must be UTC timestamp")
        if spec["frequency"] != "D" or spec["gaps"] not in {"allow", "error"}:
            raise ValueError("Initial version supports daily frequency, error/allow gaps")
        if type(spec["target_delay_periods"]) is not int or spec["target_delay_periods"] < 0:
            raise ValueError("Explicit nonnegative target delay required")
        if spec["origin_policy"] not in {"start_of_business_day", "end_of_business_day"}:
            raise ValueError("Explicit business-day origin policy required")
        return cls(spec)

    @property
    def columns(self):
        return sorted(self.spec["columns"])

    @property
    def order(self):
        return [*self.spec["series"], self.spec["time"]]

    @property
    def roles(self):
        return {
            "id": [self.spec["observation_id"]],
            "series": self.spec["series"],
            "time": [self.spec["time"]],
            "target": [self.spec["target"]],
            "feature": self.spec["features"],
            "known_future": self.spec["known_future"],
            "available_at": [self.spec["available_at"]] if self.spec.get("available_at") else [],
        }

    def parse(self, record):
        if set(record) != set(self.columns):
            raise ValueError("Observation columns do not match schema")
        values = []
        for col in self.columns:
            value, kind = (record[col], self.spec["columns"][col])
            if value is None or value == "":
                if col not in self.spec["nullable"]:
                    raise ValueError(f"Null in required field: {col}")
                values.append(None)
                continue
            text = str(value)
            if kind == "int":
                if not re.fullmatch("[+-]?\\d+", text):
                    raise ValueError(f"Invalid integer: {col}")
                value = int(text)
                if not -(2**63) <= value < 2**63:
                    raise ValueError("Integer outside signed 64-bit range")
            elif kind == "float":
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite number")
                if value == 0:
                    value = 0.0
            elif kind == "date":
                value = checked_date(text)
            elif kind == "utc_timestamp":
                value = checked_utc(text)
            else:
                value = text
            values.append(value)
        return tuple(values)

    def new_hasher(self):
        h = hashlib.sha256()
        h.update((canonical({"format": HASH_FORMAT, "columns": self.columns}) + "\n").encode())
        return h

    def arrow_schema(self):
        import pyarrow as pa

        types = {
            "string": pa.string(),
            "int": pa.int64(),
            "float": pa.float64(),
            "date": pa.date32(),
            "utc_timestamp": pa.timestamp("us", tz="UTC"),
        }
        return pa.schema(
            [
                pa.field(c, types[self.spec["columns"][c]], nullable=c in self.spec["nullable"])
                for c in self.columns
            ]
        )

    def arrow_table(self, records):
        import pyarrow as pa

        schema = self.arrow_schema()
        arrays = []
        for index, col in enumerate(self.columns):
            vals = [row[index] for row in records]
            kind = self.spec["columns"][col]
            if kind == "date":
                vals = [date.fromisoformat(v) if v is not None else None for v in vals]
            elif kind == "utc_timestamp":
                vals = [
                    datetime.fromisoformat(v.replace("Z", "+00:00")) if v is not None else None
                    for v in vals
                ]
            arrays.append(pa.array(vals, type=schema.field(col).type))
        return pa.Table.from_arrays(arrays, schema=schema)
