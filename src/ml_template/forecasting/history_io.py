"Verified plan/Silver adapter. No model or metrics execute during prediction."

from .history import recipe
from .history_engine import _predict, _score


def context(plan_path, silver_dataset):
    from ml_template.temporal_stream.silver_io import load_temporal_silver

    from .io import verify_silver_binding

    plan = verify_silver_binding(plan_path, silver_dataset)
    silver = load_temporal_silver(silver_dataset)
    if any((silver.manifest.get(k) != v for k, v in plan["identity"]["silver"].items())):
        raise ValueError("Silver changed during context loading")
    return (plan, silver)


def requests(plan_path, plan):
    from .artifacts import iter_plan_rows, verify_plan

    yield from iter_plan_rows(plan_path)
    if verify_plan(plan_path)["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("Plan changed while loading requests")


def records(silver, columns, end, batch_size, start=None):
    for batch in silver.scan_batches(
        columns=columns, start_date=start, end_date=end, batch_size=batch_size, ordered=True
    ):
        yield from batch.to_pylist()
    from ml_template.temporal_stream.silver_manifest import verify_inventory

    if verify_inventory(silver.root)["manifest_sha256"] != silver.manifest["manifest_sha256"]:
        raise ValueError("Silver changed while scanning")


def predict(plan_path, silver_dataset, configuration, output_root, database, batch_size=5000):
    plan, silver = context(plan_path, silver_dataset)
    t = silver.schema.spec
    columns = [t["observation_id"], *t["series"], t["time"], t["target"]]
    if t.get("available_at"):
        columns.append(t["available_at"])
    end = max(window["refit_end"] for window in plan["identity"]["windows"])
    if end >= plan["identity"]["holdout"]["start"]:
        raise ValueError("History overlaps reserved holdout")
    return _predict(
        plan,
        requests(plan_path, plan),
        records(silver, columns, end, batch_size),
        recipe(configuration),
        output_root,
        database,
        batch_size,
    )


def score(
    plan_path,
    silver_dataset,
    prediction_path,
    configuration,
    output_root,
    database,
    batch_size=5000,
):
    plan, silver = context(plan_path, silver_dataset)
    t = silver.schema.spec
    columns = [t["observation_id"], *t["series"], t["time"], t["target"]]
    dates = [window["target_date"] for window in plan["identity"]["windows"]]
    low, high = (min(dates), max(dates))
    if high >= plan["identity"]["holdout"]["start"]:
        raise ValueError("Scoring overlaps reserved holdout")
    return _score(
        plan,
        requests(plan_path, plan),
        prediction_path,
        records(silver, columns, high, batch_size, low),
        recipe(configuration),
        output_root,
        database,
    )
