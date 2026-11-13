"""Block ``generic_interactions``: row-wise transforms and crosses, driven by parameters.

Stateless and dataset-independent. Nothing is learned from data, so the result is stored once
like any other block. Parameters (all lists are optional)::

    transforms:        [{columns: [torque_nm], ops: [square, cube, sqrt, log, log1p,
                                                      signed_log, reciprocal]}]
    numeric_pairs:     [{a: torque_nm, b: rotational_speed_rpm, ops: [product, ratio, difference]}]
    numeric_category:  [{numeric: torque_nm, category: type, levels: [L, M, H]}]
    category_pairs:    [{a: type, b: shift}]
    nullable_outputs:  [torque_nm__minus__rotational_speed_rpm]

Domain rules are strict: log of a non-positive value, a zero denominator and similar raise
instead of producing NaN or infinity. Existing missing inputs propagate to explicitly
declared nullable outputs. Numeric/category gates propagate missingness from either input;
category crosses preserve actual nulls rather than creating string labels such as "nan".

This module also holds ``PowerTransformStep`` (Box-Cox, Yeo-Johnson). It must be fitted on
training rows, so it runs inside the training folds (see ``features/fitted_features.py``).
Note that any monotone single-column transform leaves tree-model splits unchanged; use
transforms for crosses, sums and ratios, or for models that are not tree based.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ml_template.features.block_registry import register_block

DEFAULTS: dict[str, Any] = {
    "transforms": [],
    "numeric_pairs": [],
    "numeric_category": [],
    "category_pairs": [],
    "nullable_outputs": [],
}


def _need(ok: pd.Series, message: str, column: str) -> None:
    if not bool(ok.all()):
        raise ValueError(f"{message} (column {column!r}, {int((~ok).sum())} rows)")


def _sqrt(s: pd.Series, col: str) -> pd.Series:
    _need(s.isna() | (s >= 0), "sqrt needs values >= 0", col)
    return np.sqrt(s)


def _log(s: pd.Series, col: str) -> pd.Series:
    _need(s.isna() | (s > 0), "log needs values > 0", col)
    return np.log(s)


def _log1p(s: pd.Series, col: str) -> pd.Series:
    _need(s.isna() | (s > -1), "log1p needs values > -1", col)
    return np.log1p(s)


def _reciprocal(s: pd.Series, col: str) -> pd.Series:
    _need(s.isna() | (s != 0), "reciprocal needs non-zero values", col)
    return 1.0 / s


UNARY: dict[str, Callable[[pd.Series, str], pd.Series]] = {
    "square": lambda s, c: s**2,
    "cube": lambda s, c: s**3,
    "sqrt": _sqrt,
    "log": _log,
    "log1p": _log1p,
    "signed_log": lambda s, c: np.sign(s) * np.log1p(np.abs(s)),
    "reciprocal": _reciprocal,
}


def _ratio(a: pd.Series, b: pd.Series, cb: str) -> pd.Series:
    _need(b.isna() | (b != 0), "ratio needs a non-zero denominator", cb)
    return a / b


PAIR: dict[str, tuple[str, Callable[[pd.Series, pd.Series, str], pd.Series]]] = {
    "product": ("x", lambda a, b, cb: a * b),
    "ratio": ("div", _ratio),
    "difference": ("minus", lambda a, b, cb: a - b),
}


def required_columns(params: Mapping[str, Any]) -> tuple[str, ...]:
    cols: list[str] = []
    for t in params.get("transforms", []):
        cols += list(t["columns"])
    for p in params.get("numeric_pairs", []):
        cols += [p["a"], p["b"]]
    for g in params.get("numeric_category", []):
        cols += [g["numeric"], g["category"]]
    for c in params.get("category_pairs", []):
        cols += [c["a"], c["b"]]
    return tuple(dict.fromkeys(cols))


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"column {col!r} must be numeric")
    values = df[col].astype(float)
    if not np.isfinite(values.dropna().to_numpy()).all():
        raise ValueError(f"column {col!r} contains infinite values")
    return values


def _ops(spec_ops: Sequence[str], allowed: Mapping[str, Any], what: str) -> list[str]:
    unknown = sorted(set(spec_ops) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {what} ops {unknown}; available: {sorted(allowed)}")
    return list(spec_ops)


@register_block(
    "generic_interactions",
    requires=required_columns,
    description="parameter-driven transforms, numeric/categorical crosses (stateless)",
    params=DEFAULTS,
    nullable_outputs=lambda params: params["nullable_outputs"],
)
def generic_interactions(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series) -> None:
        if name in out:
            raise ValueError(f"duplicate output column {name!r}")
        out[name] = values

    for t in params["transforms"]:
        for col in t["columns"]:
            s = _numeric(df, col)
            for op in _ops(t["ops"], UNARY, "transform"):
                add(f"{col}__{op}", UNARY[op](s, col))
    for p in params["numeric_pairs"]:
        a, b = _numeric(df, p["a"]), _numeric(df, p["b"])
        for op in _ops(p["ops"], PAIR, "pair"):
            tag, fn = PAIR[op]
            add(f"{p['a']}__{tag}__{p['b']}", fn(a, b, p["b"]))
    for g in params["numeric_category"]:
        s, cat = _numeric(df, g["numeric"]), df[g["category"]].astype("string")
        for level in g["levels"]:
            values = s.where(cat.eq(str(level)).fillna(False), 0.0)
            values = values.mask(s.isna() | cat.isna())
            add(f"{g['numeric']}__if_{g['category']}_{level}", values)
    for c in params["category_pairs"]:
        add(
            f"{c['a']}__x__{c['b']}",
            df[c["a"]].astype("string") + "|" + df[c["b"]].astype("string"),
        )
    if not out:
        raise ValueError("generic_interactions has nothing to build; set at least one section")
    return pd.DataFrame(out, index=df.index)


class PowerTransformStep:
    """Box-Cox or Yeo-Johnson on chosen columns, fitted on training rows only."""

    name = "power_transform"

    def __init__(
        self,
        columns: Sequence[str],
        method: str = "yeo-johnson",
        standardize: bool = False,
        prefix: str = "pt_",
    ) -> None:
        if method not in {"box-cox", "yeo-johnson"}:
            raise ValueError("method must be 'box-cox' or 'yeo-johnson'")
        if not columns:
            raise ValueError("power_transform needs at least one column")
        self.columns = list(columns)
        self.method = method
        self.standardize = standardize
        self.prefix = prefix

    def output_columns(self) -> list[str]:
        return [f"{self.prefix}{c}" for c in self.columns]

    def _data(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X[self.columns].astype(float)
        if self.method == "box-cox" and not bool((data > 0).all().all()):
            raise ValueError("box-cox needs strictly positive values; use yeo-johnson instead")
        return data

    def fit(self, X: pd.DataFrame) -> PowerTransformStep:
        from sklearn.preprocessing import PowerTransformer

        self.model_ = PowerTransformer(method=self.method, standardize=self.standardize)
        self.model_.fit(self._data(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self.model_.transform(self._data(X))
        if not np.isfinite(values).all():
            raise ValueError("power_transform produced non-finite values")
        return pd.DataFrame(values, index=X.index, columns=self.output_columns())
