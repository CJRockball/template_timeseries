"""Ingest raw data files into SQLite (bronze layer).

Raw copy with lineage: adds dataset_version_id and source_row_number.
Registers every raw column in column_registry and logs ingestion event
in ingestion_log.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import (
    Engine,
    text,
)

from ml_template.data.inspect_bronze import (
    SourceStructuralSchema,
    inspect_source_schema,
    iter_batches,
    validate_bronze_contract,
)
from ml_template.data.schema import Status
from ml_template.db.bronze_tables import setup_db
from ml_template.db.connection import get_engine
from ml_template.tracking.git_state import git_sha
from ml_template.tracking.pipeline_logging import setup_logging

logger = logging.getLogger("ml_template.data.ingest")


def _to_canonical_name(raw_name: str) -> str:
    """Sanitize a raw column header into a canonical snake_case identifier."""
    # Remove brackets, parentheses, and special punctuation
    clean = re.sub(r"[\[\]\(\)\{\}\<\>]", "", raw_name)
    # Replace spaces, hyphens, and non-alphanumeric chars with underscores
    clean = re.sub(r"[\s\-\.]+", "_", clean.strip().lower())
    # Strip any consecutive or trailing underscores
    return re.sub(r"_+", "_", clean).strip("_")


def build_registry(
    schema: SourceStructuralSchema,
    dataset_version_id: str,
    contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Construct column_registry entries for a dataset version.

    Supports semantic grouping lists (id, features, target, outcomes) directly
    from the bronze_source_contract.
    """
    contract_data = contract or {}
    canonical_overrides = contract_data.get("canonical_names", {})
    descriptions = contract_data.get("column_descriptions", {})
    required_cols = set(
        contract_data.get("required_columns", contract_data.get("col_names", schema.columns))
    )

    # 1. Resolve semantic roles from contract lists
    # Supports both grouped `roles: {id: [...], feature: [...]}`
    # and direct lists `id: [...]`, `features: [...]`, `target: [...]`, `outcomes: [...]`
    column_role_map: dict[str, str] = {}

    roles_dict = contract_data.get("roles", {})
    if roles_dict:
        for role_name, col_list in roles_dict.items():
            for col in col_list:
                column_role_map[col] = role_name.lower().rstrip(
                    "s"
                )  # e.g., 'features' -> 'feature'

    # Fallback to direct contract keys if not under `roles:`
    role_key_mapping = {
        "id": "id",
        "features": "feature",
        "feature": "feature",
        "target": "target",
        "outcomes": "outcome",
        "outcome": "outcome",
    }
    for key, role_label in role_key_mapping.items():
        if key in contract_data and isinstance(contract_data[key], list):
            for col in contract_data[key]:
                # Do not overwrite if already explicitly mapped
                if col not in column_role_map or role_label == "target":
                    column_role_map[col] = role_label

    now_utc = datetime.now(UTC)
    records: list[dict[str, Any]] = []

    for idx, col in enumerate(schema.columns):
        # Determine semantic role
        role = column_role_map.get(col, "unknown")

        # Determine canonical name
        canonical_name = canonical_overrides.get(col, _to_canonical_name(col))

        records.append(
            {
                "dataset_version_id": dataset_version_id,
                "column_name": col,
                "canonical_column_name": canonical_name,
                "source_position": idx,
                "source_dtype": schema.column_types.get(col, "string"),
                "semantic_role": role,
                "is_required": col in required_cols,
                "description": descriptions.get(col, None),
                "created_at": now_utc,
            }
        )

    return pd.DataFrame(records)


def insert_batches(
    engine: Engine,
    schema: SourceStructuralSchema,
    registry: pd.DataFrame,
    dataset_version_id: str,
    git_sha: str,
    started_at: datetime,
    batch_size: int = 50_000,
) -> int:
    """Stream raw batches into SQLite and record lineage and audit records.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy database engine.
    schema : SourceStructuralSchema
        Discovered technical schema of the source file (contains source_sha256).
    registry : pd.DataFrame
        Prepared column_registry DataFrame from build_registry.
    dataset_version_id : str
        Unique identifier for this dataset version snapshot.
    git_sha : str
        Git commit SHA of the current working tree.
    started_at : datetime
        UTC timestamp marking when ingestion began.
    batch_size : int, optional
        Target chunk row count (default 50,000).

    Returns
    -------
    int
        Total number of rows ingested into raw_records.
    """
    total_rows = 0
    source_row_offset = 0

    # Execute inside a single transactional block for atomicity.
    # If an exception occurs, the entire transaction rolls back.
    with engine.begin() as conn:
        # 1. Idempotency safeguard: purge existing records for THIS version if re-running
        for table_name in ("raw_records", "column_registry", "ingestion_log"):
            conn.execute(
                text(f"DELETE FROM {table_name} WHERE dataset_version_id = :dvid"),
                {"dvid": dataset_version_id},
            )

        # 2. Stream chunks and write directly to raw_records
        logger.info(
            "Streaming batches from %s into raw_records (version: %s)...",
            schema.source_path.name,
            dataset_version_id,
        )

        for batch_df in iter_batches(schema=schema, batch_size=batch_size):
            num_rows = len(batch_df)
            if num_rows == 0:
                continue

            # Assign lineage attributes
            row_numbers = range(source_row_offset + 1, source_row_offset + num_rows + 1)
            batch_df.insert(0, "source_row_number", row_numbers)
            batch_df.insert(0, "dataset_version_id", dataset_version_id)

            # Insert batch into raw_records
            batch_df.to_sql(
                "raw_records",
                conn,
                if_exists="append",
                index=False,
                chunksize=2_000,
            )

            source_row_offset += num_rows
            total_rows += num_rows

        # 3. Insert prepared column_registry
        registry.to_sql(
            "column_registry",
            conn,
            if_exists="append",
            index=False,
        )

        # 4. Construct and insert single ingestion_log record
        completed_at = datetime.now(UTC)
        source_path_val = str(schema.source_path)
        source_name_val = schema.source_path.name

        log_entry = {
            "dataset_version_id": dataset_version_id,
            "source_name": source_name_val,
            "source_format": schema.format,
            "source_path_or_uri": source_path_val,
            "source_sha256": schema.source_sha256,
            "schema_hash": schema.schema_hash,
            "row_count_read": total_rows,
            "row_count_written": total_rows,
            "columns_count": len(schema.columns),
            "encoding": schema.encoding,
            "delimiter": schema.delimiter,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": Status.SUCCEEDED.value,
            "error_message": None,
            "git_sha": git_sha,
        }

        log_df = pd.DataFrame([log_entry])
        log_df.to_sql(
            "ingestion_log",
            conn,
            if_exists="append",
            index=False,
        )

    logger.info(
        "Successfully ingested %d rows, %d columns into bronze raw_records (version: %s)",
        total_rows,
        len(schema.columns),
        dataset_version_id,
    )
    return total_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingest raw data into bronze SQLite database")
    p.add_argument(
        "--contract_file",
        type=str,
        required=True,
        help="Path to data to be ingested",
    )
    p.add_argument(
        "--dataset_version_id",
        type=str,
        default="ds_v1",
        help="Identifier for this dataset version (default: ds_v1)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Sets up console + logs/app/pipeline.log
    setup_logging()
    logger.info("Started raw data ingestion")

    # 1. Capture execution metadata
    started_at = datetime.now(UTC)
    current_git_sha = git_sha()
    dataset_version_id = args.dataset_version_id

    # 2. Load configurations
    conf_path = Path(args.contract_file)
    with conf_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    source_path = Path(data_config["data"]["source"])
    source_format = data_config["data"].get("format")

    # 3. Discover schema and comput source_sha256
    schema = inspect_source_schema(source_path, source_format)

    # 4. Enforce structural bronze contract
    contract = data_config.get("bronze_source_contract", {})
    validate_bronze_contract(schema, contract)

    # 5. Initialize database tables dynamically
    engine = get_engine()
    setup_db(engine, schema)

    # 6. Build column registry metadata
    registry = build_registry(
        schema,
        dataset_version_id,
        contract,
    )

    # 7 Stream batches and write to DB atomatically
    insert_batches(engine, schema, registry, dataset_version_id, current_git_sha, started_at)

    logger.info("Completed successfully")


if __name__ == "__main__":
    main()
