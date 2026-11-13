"""Extraction functions for bronze data."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sqlalchemy import Engine, text

from ml_template.db.connection import get_engine

logger = logging.getLogger("ml_template.data.extract")


def extract_bronze(
    dataset_version_id: str,
    columns: list[str] | None = None,
    sample_n: int | None = None,
    sampling: Literal["contiguous", "random", "all"] = "contiguous",
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Extract records for a specific dataset version from the bronze layer.

    Parameters
    ----------
    dataset_version_id : str
        Unique version identifier recorded in raw_records.
    columns : list[str] | None, default None
        Specific columns to select. If None, all columns are retrieved.
    sample_n : int | None, default None
        Maximum row count to return. If None, all available rows are returned.
    sampling : {"contiguous", "random", "all"}, default "contiguous"
        Sampling strategy when sample_n is specified:
        - "contiguous": Sequential extraction ordered by source_row_number (ideal for time-series).
        - "random": Uniform pseudo-random row extraction.
        - "all": Ignores sampling limit and retrieves all rows.
    engine : Engine | None, default None
        SQLAlchemy engine instance. If None, uses default project engine.

    Returns
    -------
    pd.DataFrame
        Extracted bronze dataset records.
    """
    db_engine = engine or get_engine()

    # Quote column identifiers to handle special characters (e.g., "[K]", spaces)
    quoted_cols = ", ".join(f'"{col}"' for col in columns) if columns is not None else "*"

    query_parts = [
        f"SELECT {quoted_cols} FROM raw_records",
        "WHERE dataset_version_id = :dataset_version_id",
    ]
    params: dict[str, str | int] = {"dataset_version_id": dataset_version_id}

    if sample_n is not None and sampling != "all":
        if sampling == "random":
            query_parts.append("ORDER BY RANDOM() LIMIT :limit")
        elif sampling == "contiguous":
            query_parts.append("ORDER BY source_row_number ASC LIMIT :limit")
        else:
            msg = f"Unsupported sampling strategy: '{sampling}'. Choose 'contiguous', 'random', or 'all'."
            raise ValueError(msg)
        params["limit"] = sample_n
    else:
        query_parts.append("ORDER BY source_row_number ASC")

    query_sql = " ".join(query_parts)

    with db_engine.connect() as conn:
        df = pd.read_sql(text(query_sql), conn, params=params)

    if df.empty:
        logger.warning(
            "Extraction returned 0 rows for dataset_version_id='%s'",
            dataset_version_id,
        )
    else:
        logger.info(
            "Extracted %d rows and %d columns from bronze (version: '%s', strategy: '%s')",
            len(df),
            df.shape[1],
            dataset_version_id,
            sampling if sample_n else "full",
        )

    return df
