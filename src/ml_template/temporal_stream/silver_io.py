"Batch-oriented readers and round-trip verification for temporal Silver."

from datetime import date, datetime
from pathlib import Path

from .silver_manifest import verify_inventory
from .silver_schema import TemporalSchema, canonical, checked_date, positive


def canonical_record(schema, record):
    out = {}
    for col in schema.columns:
        value = record[col]
        if isinstance(value, datetime):
            value = value.isoformat(timespec="microseconds").replace("+00:00", "Z")
        elif isinstance(value, date):
            value = value.isoformat()
        out[col] = value
    return schema.parse(out)


def validate_parquet(root, manifest, batch_size=5000):
    import pyarrow.parquet as pq

    positive(batch_size, "batch_size")
    root = Path(root)
    schema = TemporalSchema.from_dict(manifest["temporal_schema"])
    expected = schema.arrow_schema()
    h = schema.new_hasher()
    total = 0
    previous = None
    indices = [schema.columns.index(c) for c in schema.order]
    for item in manifest["files"]:
        with pq.ParquetFile(root / item["path"]) as file:
            if (
                not file.schema_arrow.equals(expected, check_metadata=False)
                or file.metadata.num_rows != item["rows"]
            ):
                raise ValueError("Parquet schema/count mismatch")
            for batch in file.iter_batches(
                batch_size=batch_size, columns=schema.columns, use_threads=False
            ):
                for record in batch.to_pylist():
                    row = canonical_record(schema, record)
                    key = tuple(row[i] for i in indices)
                    if previous is not None and key <= previous:
                        raise ValueError("Non-increasing temporal row order")
                    previous = key
                    h.update((canonical(list(row)) + "\n").encode())
                    total += 1
    if total != manifest["row_count"] or h.hexdigest() != manifest["logical_data_sha256"]:
        raise ValueError("Parquet logical round-trip integrity failure")
    return total


class TemporalSilver:
    """Verified dataset handle; deliberately has no implicit model_frame/to_pandas."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = verify_inventory(self.root)
        self.schema = TemporalSchema.from_dict(self.manifest["temporal_schema"])

    def scan_batches(
        self,
        *,
        columns=None,
        start_date=None,
        end_date=None,
        series_values=None,
        batch_size=5000,
        ordered=True,
        batch_readahead=0,
        fragment_readahead=0,
    ):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        positive(batch_size, "batch_size")
        for value in [batch_readahead, fragment_readahead]:
            if type(value) is not int or value < 0:
                raise ValueError("Readahead must be a nonnegative integer")
        current = verify_inventory(self.root)
        if current["manifest_sha256"] != self.manifest["manifest_sha256"]:
            raise ValueError("Dataset manifest changed after opening")
        columns = self.schema.columns if columns is None else list(columns)
        if (
            not columns
            or len(columns) != len(set(columns))
            or (not set(columns) <= set(self.schema.columns))
        ):
            raise ValueError("Select unique declared columns")
        if start_date is not None:
            start_date = date.fromisoformat(checked_date(start_date))
        if end_date is not None:
            end_date = date.fromisoformat(checked_date(end_date))
        if start_date and end_date and (start_date > end_date):
            raise ValueError("start_date exceeds end_date")
        series_values = series_values or {}
        if not set(series_values) <= set(self.schema.spec["series"]):
            raise ValueError("Unknown series filter column")
        schema = self.schema.arrow_schema()
        for col, values in series_values.items():
            if not isinstance(values, (list, tuple)) or any(v is None for v in values):
                raise ValueError("Series filter must be a non-null list/tuple")
            pa.array(values, type=schema.field(col).type)
        time = self.schema.spec["time"]
        files = [str(self.root / item["path"]) for item in self.manifest["files"]]
        if not ordered:
            expression = None
            terms = []
            if start_date:
                terms.append(ds.field(time) >= pa.scalar(start_date, type=pa.date32()))
            if end_date:
                terms.append(ds.field(time) <= pa.scalar(end_date, type=pa.date32()))
            terms.extend(
                (
                    ds.field(col).isin(pa.array(values, type=schema.field(col).type))
                    for col, values in series_values.items()
                )
            )
            for term in terms:
                expression = term if expression is None else expression & term
            scanner = ds.dataset(files, format="parquet", schema=schema).scanner(
                columns=columns,
                filter=expression,
                batch_size=batch_size,
                use_threads=False,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
            )
            yield from scanner.to_batches()
            return
        needed = sorted(set(columns) | {time} | set(series_values))
        for path in files:
            with pq.ParquetFile(path) as file:
                for batch in file.iter_batches(
                    batch_size=batch_size, columns=needed, use_threads=False
                ):
                    mask = None
                    conditions = []
                    values = batch.column(batch.schema.get_field_index(time))
                    if start_date:
                        conditions.append(
                            pc.greater_equal(values, pa.scalar(start_date, type=pa.date32()))
                        )
                    if end_date:
                        conditions.append(
                            pc.less_equal(values, pa.scalar(end_date, type=pa.date32()))
                        )
                    for col, vals in series_values.items():
                        conditions.append(
                            pc.is_in(
                                batch.column(batch.schema.get_field_index(col)),
                                value_set=pa.array(vals, type=schema.field(col).type),
                            )
                        )
                    for condition in conditions:
                        mask = condition if mask is None else pc.and_(mask, condition)
                    if mask is not None:
                        batch = batch.filter(pc.fill_null(mask, False))
                    if batch.num_rows:
                        yield batch.select(columns)


def load_temporal_silver(root):
    return TemporalSilver(root)
