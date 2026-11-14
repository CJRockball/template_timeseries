"The only Silver bridge: projects identifiers, dates and target availability only."

from pathlib import Path

from .planner import _build_plan


def build_from_silver(silver_dataset, policy, database, output_root, batch_size=5000):
    from ml_template.temporal_stream.silver_io import load_temporal_silver

    silver = load_temporal_silver(silver_dataset)
    manifest = silver.manifest
    schema = silver.schema.spec
    for col in [schema["observation_id"], *schema["series"]]:
        if schema["columns"][col] not in {"string", "int"}:
            raise ValueError("Origin v1 requires string/integer identity columns")
    columns = [schema["observation_id"], *schema["series"], schema["time"]]
    if schema.get("available_at"):
        columns.append(schema["available_at"])
    binding = {
        key: manifest[key]
        for key in [
            "silver_dataset_version_id",
            "manifest_sha256",
            "logical_data_sha256",
            "row_count",
            "temporal_schema",
        ]
    }

    def records():
        for batch in silver.scan_batches(columns=columns, batch_size=batch_size, ordered=True):
            yield from batch.to_pylist()
        from ml_template.temporal_stream.silver_manifest import verify_inventory

        current = verify_inventory(Path(silver_dataset))
        if current["manifest_sha256"] != binding["manifest_sha256"]:
            raise ValueError("Silver changed during origin planning")

    return _build_plan(records(), binding, policy, output_root, database, batch_size)


def verify_silver_binding(plan_path, silver_dataset):
    from ml_template.temporal_stream.silver_io import load_temporal_silver

    from .artifacts import verify_plan

    plan = verify_plan(plan_path)
    current = load_temporal_silver(silver_dataset).manifest
    expected = plan["identity"]["silver"]
    if any((current.get(key) != value for key, value in expected.items())):
        raise ValueError("Plan belongs to a different Silver snapshot or contract")
    return plan


def inspect_calendar(silver_dataset, holdout_periods=28, batch_size=5000):
    from datetime import timedelta

    from ml_template.temporal_stream.silver_io import load_temporal_silver

    from .policy import day

    if type(holdout_periods) is not int or holdout_periods < 1:
        raise ValueError("Positive holdout period count required")
    silver = load_temporal_silver(silver_dataset)
    time = silver.schema.spec["time"]
    first, last, count = (None, None, 0)
    for batch in silver.scan_batches(columns=[time], batch_size=batch_size, ordered=True):
        for value in batch.column(0).to_pylist():
            value = day(value)
            first = value if first is None else min(first, value)
            last = value if last is None else max(last, value)
            count += 1
    if count != silver.manifest["row_count"] or first is None:
        raise ValueError("Invalid Silver calendar scan")
    start = last - timedelta(days=holdout_periods - 1)
    end_of_day = silver.schema.spec["origin_policy"] == "end_of_business_day"
    return {
        "first_date": first.isoformat(),
        "last_date": last.isoformat(),
        "row_count": count,
        "reserved_holdout_start": start.isoformat(),
        "reserved_holdout_end": last.isoformat(),
        "latest_development_origin": (start - timedelta(days=2 if end_of_day else 1)).isoformat(),
        "target_values_projected": False,
    }
