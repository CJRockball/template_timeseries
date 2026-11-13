from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq

# Default batch size in rows (roughly 50,000 rows ~ 25-50 MB memory)
DEFAULT_BATCH_SIZE = 50_000


@dataclass(frozen=True)
class SourceStructuralSchema:
    """Discovered technical metadata of the raw source file."""

    source_path: Path
    format: str  # "csv" or "parquet"
    columns: list[str]
    column_types: dict[str, str]  # col_name -> technical type string
    schema_hash: str
    source_sha256: str
    encoding: str | None = None
    delimiter: str | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_schema_hash(columns: list[str], column_types: dict[str, str]) -> str:
    """Compute a deterministic SHA-256 fingerprint of the ordered columns and types."""
    fingerprint = ";".join(f"{col}:{column_types.get(col, 'string')}" for col in columns)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def inspect_source_schema(
    source_path: Path,
    source_format: str | None = None,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> SourceStructuralSchema:
    """Inspect the source file to extract column headers and technical storage types.

    For CSV:
        - Reads the header row using utf-8-sig to strip BOM automatically.
        - Bronze treats all raw CSV columns as 'string' to preserve exact source bytes.

    For Parquet:
        - Reads schema metadata directly without reading data into memory.
        - Preserves native Arrow data types.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Determine format if not explicitly provided
    if source_format is None:
        suffix = source_path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            source_format = "csv"
        elif suffix in {".parquet", ".pq"}:
            source_format = "parquet"
        else:
            raise ValueError(
                f"Unsupported source format for '{source_path.name}'. "
                "Explicitly configure format as 'csv' or 'parquet'."
            )

    source_format = source_format.lower()
    source_sha256 = sha256_file(source_path)

    if source_format == "csv":
        read_encoding = (
            "utf-8-sig" if encoding.lower() in {"utf-8", "utf8", "utf-8-sig"} else encoding
        )

        with source_path.open("r", encoding=read_encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError(f"Source CSV file is empty: {source_path}") from None

        headers = [h.strip().lstrip("\ufeff") for h in headers]
        if not headers or all(len(h) == 0 for h in headers):
            raise ValueError(f"No valid headers found in source CSV: {source_path}")

        seen = set()
        duplicates = [h for h in headers if h in seen or seen.add(h)]
        if duplicates:
            raise ValueError(f"Duplicate column headers found in source CSV: {duplicates}")

        column_types = {col: "string" for col in headers}
        schema_hash = compute_schema_hash(headers, column_types)

        return SourceStructuralSchema(
            source_path=source_path,
            format="csv",
            columns=headers,
            column_types=column_types,
            schema_hash=schema_hash,
            source_sha256=source_sha256,
            encoding=encoding,
            delimiter=delimiter,
        )

    elif source_format == "parquet":
        parquet_file = pq.ParquetFile(source_path)
        arrow_schema = parquet_file.schema_arrow

        headers = [name.lstrip("\ufeff") for name in arrow_schema.names]
        column_types = {
            name: str(arrow_schema.field(raw_name).type)
            for name, raw_name in zip(headers, arrow_schema.names, strict=True)
        }
        schema_hash = compute_schema_hash(headers, column_types)

        return SourceStructuralSchema(
            source_path=source_path,
            format="parquet",
            columns=headers,
            column_types=column_types,
            schema_hash=schema_hash,
            source_sha256=source_sha256,
            encoding=None,
            delimiter=None,
        )

    else:
        raise ValueError(f"Unsupported format: {source_format}")


def validate_bronze_contract(
    schema: SourceStructuralSchema,
    contract: dict | None = None,
) -> None:
    """Validate discovered source schema against the minimal bronze structural contract."""
    if not contract:
        return

    if "columns" in contract:
        expected_n_cols = contract["columns"]
        actual_n_cols = len(schema.columns)
        if actual_n_cols != expected_n_cols:
            raise AssertionError(
                f"Bronze structural check failed: expected {expected_n_cols} columns, "
                f"but found {actual_n_cols} in {schema.source_path.name}"
            )

    if "col_names" in contract:
        expected_cols = set(contract["col_names"])
        actual_cols = set(schema.columns)
        missing = expected_cols - actual_cols
        if missing:
            raise AssertionError(
                f"Bronze structural check failed: missing required column(s) {sorted(missing)} "
                f"in {schema.source_path.name}"
            )

    if "required_columns" in contract:
        required = set(contract["required_columns"])
        actual_cols = set(schema.columns)
        missing_required = required - actual_cols
        if missing_required:
            raise AssertionError(
                f"Bronze structural check failed: missing mandatory column(s) {sorted(missing_required)} "
                f"in {schema.source_path.name}"
            )


def _iter_csv_batches(
    source_path: Path,
    schema: SourceStructuralSchema,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """Stream CSV chunks using PyArrow streaming CSV reader.

    All columns are read strictly as strings/text to preserve raw fidelity.
    """
    # Force all discovered columns to string type in Arrow
    import pyarrow as pa

    convert_options = pv_csv.ConvertOptions(
        column_types={col: pa.string() for col in schema.columns},
        strings_can_be_null=True,
    )

    read_options = pv_csv.ReadOptions(
        block_size=10 * 1024 * 1024,  # 10 MB I/O read block
        encoding=schema.encoding or "utf-8",
    )

    parse_options = pv_csv.ParseOptions(
        delimiter=schema.delimiter or ",",
    )

    reader = pv_csv.open_csv(
        source_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    # Accumulate arrow batches until reaching batch_size rows
    accumulated_batches = []
    current_rows = 0

    for batch in reader:
        accumulated_batches.append(batch)
        current_rows += batch.num_rows

        if current_rows >= batch_size:
            table = pa.Table.from_batches(accumulated_batches)
            yield table.to_pandas()
            accumulated_batches = []
            current_rows = 0

    if accumulated_batches:
        table = pa.Table.from_batches(accumulated_batches)
        yield table.to_pandas()


def _iter_parquet_batches(
    source_path: Path,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """Stream Parquet chunks using PyArrow ParquetFile iterator.

    Preserves native data types embedded in the Parquet file.
    """
    parquet_file = pq.ParquetFile(source_path)

    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        yield record_batch.to_pandas()


def iter_batches(
    schema: SourceStructuralSchema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield bounded batches of rows from the raw source file.

    Dispatches based on schema.format ('csv' or 'parquet').
    Yields each chunk as a pandas DataFrame containing only the raw columns.

    Parameters
    ----------
    schema : SourceStructuralSchema
        The technical metadata discovered by inspect_source_schema.
    batch_size : int, optional
        Target number of rows per batch (default 50,000).

    Yields
    ------
    Iterator[pd.DataFrame]
        A generator of raw DataFrames.
    """
    if schema.format == "csv":
        yield from _iter_csv_batches(
            source_path=schema.source_path,
            schema=schema,
            batch_size=batch_size,
        )
    elif schema.format == "parquet":
        yield from _iter_parquet_batches(
            source_path=schema.source_path,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported format for streaming: {schema.format}")
