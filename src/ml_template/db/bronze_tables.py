"""Database table definitions and DDL generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
)

if TYPE_CHECKING:
    from ml_template.data.inspect_bronze import SourceStructuralSchema

TYPE_MAP = {
    "string": String,
    "str": String,
    "object": String,
    "category": String,
    "int8": Integer,
    "int16": Integer,
    "int32": Integer,
    "int64": Integer,
    "uint8": Integer,
    "uint16": Integer,
    "uint32": Integer,
    "uint64": Integer,
    "float": Float,
    "float16": Float,
    "float32": Float,
    "float64": Float,
    "double": Float,
    "bool": Boolean,
    "boolean": Boolean,
}


def setup_db(engine: Engine, schema: SourceStructuralSchema) -> None:
    """Create bronze tables dynamically based on discovered source columns."""
    metadata = MetaData()

    raw_columns = [
        Column("dataset_version_id", String, primary_key=True),
        Column("source_row_number", Integer, primary_key=True),
    ]

    for col_name in schema.columns:
        type_str = schema.column_types.get(col_name, "string").lower()
        sql_type = TYPE_MAP.get(type_str, String)
        raw_columns.append(Column(col_name, sql_type))

    Table("raw_records", metadata, *raw_columns)

    Table(
        "column_registry",
        metadata,
        Column("dataset_version_id", String, primary_key=True),
        Column("column_name", String, primary_key=True),
        Column("canonical_column_name", String, nullable=True),
        Column("source_position", Integer, nullable=False),
        Column("source_dtype", String, nullable=False),
        Column("semantic_role", String, nullable=True),
        Column("is_required", Boolean, nullable=False, default=True),
        Column("description", String, nullable=True),
        Column("created_at", DateTime, nullable=False),
    )

    Table(
        "ingestion_log",
        metadata,
        Column("ingest_id", Integer, primary_key=True, autoincrement=True),
        Column("dataset_version_id", String, unique=True, nullable=False),
        Column("source_name", String, nullable=False),
        Column("source_format", String, nullable=False),
        Column("source_path_or_uri", String, nullable=False),
        Column("source_sha256", String, nullable=False),
        Column("schema_hash", String, nullable=False),
        Column("row_count_read", Integer, nullable=True),
        Column("row_count_written", Integer, nullable=True),
        Column("columns_count", Integer, nullable=False),
        Column("encoding", String, nullable=True),
        Column("delimiter", String, nullable=True),
        Column("started_at", DateTime, nullable=False),
        Column("completed_at", DateTime, nullable=True),
        Column("status", String, nullable=False),
        Column("error_message", String, nullable=True),
        Column("git_sha", String, nullable=True),
    )

    metadata.create_all(engine)
