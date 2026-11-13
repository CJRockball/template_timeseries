"""Frozen schemas with opt-in null preservation; unspecified columns stay strict."""

import numpy as np
import pandas as pd


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or type(value) not in (str, int, float, bool):
        raise ValueError("only scalar string/numeric/bool categories are supported")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite category")
    return value


def missing_limit(policy, column):
    policy = policy or {}
    limit = float(
        policy.get("max_rate_by_column", {}).get(column, policy.get("default_max_rate", 0.0))
    )
    if not 0 <= limit <= 1:
        raise ValueError(f"invalid missing-rate limit: {column}")
    return limit


def _check_nulls(series, limit, enforce_rates=False, *, nullable=None):
    rate = float(series.isna().mean())
    if nullable is not None:
        if type(nullable) is not bool:
            raise ValueError("nullable must be a boolean")
        if rate and not nullable:
            raise ValueError(f"missing values are unsupported: {series.name}")
        return
    # Legacy V1/V2 behavior only. V3 separates permission from observed rate.
    if not 0 <= limit <= 1:
        raise ValueError("missing-rate limit must be in [0,1]")
    if rate and limit == 0:
        raise ValueError(f"missing values are unsupported: {series.name}")
    if enforce_rates and rate > limit:
        raise ValueError(f"missing rate exceeds limit: {series.name}: {rate} > {limit}")


def frame_schema(frame, *, missingness=None, nullable=None):
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("nonempty frame with unique columns required")
    if nullable is not None and type(nullable) is not bool:
        raise ValueError("nullable must be a boolean or None for legacy policy")
    if nullable is not None and missingness is not None:
        raise ValueError("use nullable OR legacy missingness, not both")
    schema = []
    for name in frame.columns:
        series = frame[name]
        limit = missing_limit(missingness, name) if nullable is None else 0.0
        _check_nulls(series, limit, enforce_rates=True, nullable=nullable)
        if isinstance(series.dtype, pd.CategoricalDtype):
            item = {
                "kind": "category",
                "categories": [scalar(v) for v in series.cat.categories],
                "ordered": bool(series.cat.ordered),
            }
        elif pd.api.types.is_bool_dtype(series.dtype):
            item = {"kind": "bool", "dtype": "boolean" if nullable else str(series.dtype)}
        elif pd.api.types.is_integer_dtype(series.dtype):
            dtype = pd.api.types.pandas_dtype(series.dtype)
            if nullable:
                native = np.dtype(getattr(dtype, "numpy_dtype", dtype))
                prefix = "UInt" if native.kind == "u" else "Int"
                dtype = f"{prefix}{native.itemsize * 8}"
            item = {"kind": "integer", "dtype": str(dtype)}
        elif pd.api.types.is_float_dtype(series.dtype):
            item = {"kind": "float", "dtype": str(series.dtype)}
        elif pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
            series.dtype
        ):
            if not series.dropna().map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"mixed/non-string object column unsupported: {name}")
            item = {"kind": "string", "values": sorted(series.dropna().unique().tolist())}
        else:
            raise ValueError(f"unsupported input dtype: {name}: {series.dtype}")
        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not np.isfinite(series.dropna().to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"nonfinite numeric input: {name}")
        if nullable is not None:
            item.update(nullable=nullable, observed_missing_rate=float(series.isna().mean()))
        elif missingness is not None:
            item["max_missing_rate"] = limit
        schema.append({"name": str(name), **item})
    return schema


def coerce_frame(frame, schema, *, enforce_rates=False):
    names = [item["name"] for item in schema]
    if not names or len(set(names)) != len(names) or frame.columns.duplicated().any():
        raise ValueError("invalid/duplicate column schema")
    if set(frame.columns) != set(names):
        raise ValueError(
            f"column mismatch; missing={sorted(set(names) - set(frame.columns))}, "
            f"extra={sorted(set(frame.columns) - set(names))}"
        )
    if frame.empty:
        raise ValueError("empty scoring frame")
    out = frame[names].copy()
    for item in schema:
        name, kind = item["name"], item["kind"]
        series = out[name]
        _check_nulls(
            series,
            float(item.get("max_missing_rate", 0.0)),
            enforce_rates,
            nullable=item.get("nullable"),
        )
        values = series.dropna()
        if kind in {"category", "string"}:
            allowed = item["categories"] if kind == "category" else item["values"]
            if not values.isin(allowed).all():
                raise ValueError(f"unseen category/value: {name}")
            out[name] = (
                pd.Categorical(series, categories=allowed, ordered=item["ordered"])
                if kind == "category"
                else series.astype(object)
            )
        elif kind == "bool":
            if not values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
                raise ValueError(f"boolean values required: {name}")
            out[name] = series.astype(item["dtype"])
        elif kind in {"integer", "float"}:
            if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(
                series.dtype
            ):
                raise ValueError(
                    f"numeric dtype required: {name}; implicit text parsing is disabled"
                )
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"nonfinite numeric input: {name}")
            dtype = pd.api.types.pandas_dtype(item["dtype"])
            numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
            if kind == "integer":
                if (values % 1 != 0).any():
                    raise ValueError(f"fractional integer input: {name}")
                bounds = np.iinfo(numpy_dtype)
                if any(int(v) < bounds.min or int(v) > bounds.max for v in values):
                    raise ValueError(f"integer out of range: {name}")
            cast = series.astype(dtype)
            cast_values = cast.loc[series.notna()]
            if not np.isfinite(cast_values.to_numpy(dtype=float)).all():
                raise ValueError(f"numeric overflow: {name}")
            if kind == "integer" and any(
                int(a) != int(b) for a, b in zip(values, cast_values, strict=True)
            ):
                raise ValueError(f"lossy integer conversion: {name}")
            if kind == "float" and not np.array_equal(
                values.to_numpy(dtype=float), cast_values.to_numpy(dtype=float)
            ):
                raise ValueError(f"lossy numeric conversion: {name}")
            out[name] = cast
        else:
            raise ValueError(f"unsupported schema kind: {kind}")
    return out


def validate_batch(frame, contract, *, enforce_rates=False):
    # Rates are a whole-input admission rule, not a chunk/singleton prediction rule.
    key = contract["key_col"]
    if frame.columns.duplicated().any() or key not in frame:
        raise ValueError("missing key or duplicate column names")
    keys = frame[key]
    if keys.empty or keys.isna().any() or keys.duplicated().any():
        raise ValueError("keys must be nonempty, unique and non-null")
    kind = contract["key_kind"]
    if kind not in {"str", "int"}:
        raise ValueError("unsupported entity-key schema")
    ok = keys.map(
        lambda v: (
            isinstance(v, str)
            if kind == "str"
            else isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        )
    )
    if not ok.all():
        raise ValueError("entity key type differs from release")
    out = coerce_frame(
        frame.drop(columns=[key]), contract["source_schema"], enforce_rates=enforce_rates
    )
    out.index = pd.Index(keys.to_numpy(), name=key)
    return out
