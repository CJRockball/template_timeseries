"""Shared, deterministic canonicalisation. No learned encoders or imputation."""

import numpy as np
import pandas as pd


def canonicalize_frame(frame, spec, *, columns=None, passthrough=()):
    """Accept a silver_eda_contract or a frozen release input_spec.

    `columns` is explicit: training can include its target, inference must not.
    Unknown columns are rejected; declared but unused provenance can be projected.
    """
    names = spec["canonical_names"]
    if frame.columns.duplicated().any():
        raise ValueError("duplicate raw columns")
    unknown = set(frame.columns) - set(names) - set(names.values()) - set(passthrough)
    if unknown:
        raise ValueError(f"undeclared source columns: {sorted(unknown)}")
    out = frame.rename(columns=names)
    if out.columns.duplicated().any():
        raise ValueError("canonical name collision")
    columns = list(columns if columns is not None else spec["canonical_dtypes"])
    missing = set(columns) - set(out)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    extra = [c for c in passthrough if c in out and c not in columns]
    out = out[[*columns, *extra]].copy()
    for col in columns:
        mapping = spec.get("value_mappings", {}).get(col)
        if mapping is not None:
            if isinstance(mapping, list):
                mapping = {entry["source"]: entry["value"] for entry in mapping}
            invalid = out[col].notna() & ~out[col].isin(list(mapping))
            if invalid.any():
                raise ValueError(
                    f"unmapped values: {col}: {out.loc[invalid, col].unique().tolist()}"
                )
            out[col] = out[col].map(mapping)
        dtype = pd.api.types.pandas_dtype(spec["canonical_dtypes"][col])
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric = pd.to_numeric(out[col], errors="raise")
            valid = numeric.dropna()
            if not np.isfinite(valid.to_numpy(dtype=float)).all():
                raise ValueError(f"nonfinite numeric values: {col}")
            if pd.api.types.is_integer_dtype(dtype):
                bounds = np.iinfo(getattr(dtype, "numpy_dtype", dtype))
                if (valid % 1 != 0).any() or any(
                    int(v) < bounds.min or int(v) > bounds.max for v in valid
                ):
                    raise ValueError(f"fractional or out-of-range integer: {col}")
            out[col] = numeric.astype(dtype)
        else:
            # Check before casting can turn unknown categories into NA.
            if (
                isinstance(dtype, pd.CategoricalDtype)
                and dtype.categories is not None
                and not out[col].dropna().isin(dtype.categories).all()
            ):
                raise ValueError(f"unknown category: {col}")

            out[col] = out[col].astype(dtype)
        allowed = spec.get("allowed_values", {}).get(col)
        if allowed is not None and not out[col].dropna().isin(allowed).all():
            raise ValueError(f"values outside contract: {col}")
    return out
