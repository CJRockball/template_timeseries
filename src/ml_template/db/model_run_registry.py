"""Write and read model-run records. Same shape as data_run_registry.

Pass plain values, not config objects, so this module does not depend on training
code. Write the ``running`` row first and the final status last: a row that says
``succeeded`` then means the files exist.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, delete, insert, select, update

from ml_template.db.tracking_tables import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    ensure_tracking_tables,
    experiment_launches,
    model_runs,
    run_feature_importance,
    run_metrics,
    run_params,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id(prefix: str = "run") -> str:
    """Sortable and unique: run_20261011_101530_ab12cd."""
    return f"{prefix}_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _dirty_flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def ensure_tables(engine: Engine) -> None:
    ensure_tracking_tables(engine)


def start_launch(
    engine: Engine,
    *,
    experiment_name: str,
    kind: str,
    n_runs: int,
    sweep_path: str | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> str:
    ensure_tables(engine)
    launch_id = new_run_id("exp")
    with engine.begin() as conn:
        conn.execute(
            insert(experiment_launches).values(
                experiment_run_id=launch_id,
                experiment_name=experiment_name,
                kind=kind,
                sweep_path=sweep_path,
                n_runs=n_runs,
                status=STATUS_RUNNING,
                started_at=_now(),
                git_sha=git_sha,
                git_dirty=_dirty_flag(git_dirty),
            )
        )
    return launch_id


def finish_launch(
    engine: Engine,
    experiment_run_id: str,
    *,
    succeeded: bool,
    n_failed: int = 0,
    error_summary: str | None = None,
) -> None:
    with engine.begin() as conn:
        result = conn.execute(
            update(experiment_launches)
            .where(experiment_launches.c.experiment_run_id == experiment_run_id)
            .values(
                status=STATUS_SUCCEEDED if succeeded else STATUS_FAILED,
                finished_at=_now(),
                n_failed=n_failed,
                error_summary=error_summary,
            )
        )
        if result.rowcount == 0:
            raise KeyError(f"unknown experiment_run_id {experiment_run_id}")


def start_model_run(
    engine: Engine,
    *,
    run_id: str,
    experiment_run_id: str,
    experiment_name: str,
    config_hash: str,
    params: dict[str, str],
    silver_dataset_version_id: str,
    feature_set_id: str | None,
    split_id: str,
    model_type: str,
    seed: int,
    artifact_dir: str,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> None:
    ensure_tables(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(model_runs).values(
                run_id=run_id,
                experiment_run_id=experiment_run_id,
                experiment_name=experiment_name,
                config_hash=config_hash,
                status=STATUS_RUNNING,
                started_at=_now(),
                silver_dataset_version_id=silver_dataset_version_id,
                feature_set_id=feature_set_id,
                split_id=split_id,
                model_type=model_type,
                seed=seed,
                git_sha=git_sha,
                git_dirty=_dirty_flag(git_dirty),
                artifact_dir=artifact_dir,
            )
        )
        if params:
            conn.execute(
                insert(run_params),
                [{"run_id": run_id, "key": k, "value": v} for k, v in params.items()],
            )


def log_metrics(engine: Engine, run_id: str, rows: list[dict[str, Any]]) -> None:
    """Store metric rows. Rows for the same run and split are replaced, so reruns are safe."""
    if not rows:
        return
    clean = [
        {
            "run_id": run_id,
            "split": r["split"],
            "fold": int(r["fold"]),
            "metric": r["metric"],
            "value": _clean_float(r["value"]),
        }
        for r in rows
    ]
    splits_present = {r["split"] for r in clean}
    with engine.begin() as conn:
        conn.execute(
            delete(run_metrics).where(
                run_metrics.c.run_id == run_id, run_metrics.c.split.in_(splits_present)
            )
        )
        conn.execute(insert(run_metrics), clean)


def log_feature_importance(engine: Engine, run_id: str, rows: list[dict[str, Any]]) -> None:
    """Store importance rows. Rows for the same run, split and method are replaced."""
    if not rows:
        return
    clean = [
        {
            "run_id": run_id,
            "split": r["split"],
            "fold": int(r["fold"]),
            "method": r["method"],
            "feature": r["feature"],
            "value": _clean_float(r["value"]),
            "std": _clean_float(r.get("std")),
        }
        for r in rows
    ]
    keys = {(r["split"], r["method"]) for r in clean}
    with engine.begin() as conn:
        for split, method in keys:
            conn.execute(
                delete(run_feature_importance).where(
                    run_feature_importance.c.run_id == run_id,
                    run_feature_importance.c.split == split,
                    run_feature_importance.c.method == method,
                )
            )
        conn.execute(insert(run_feature_importance), clean)


def finish_model_run(
    engine: Engine,
    run_id: str,
    *,
    succeeded: bool,
    primary_metric: str | None = None,
    primary_value: float | None = None,
    n_rounds_final: int | None = None,
    n_rows: int | None = None,
    n_features: int | None = None,
    duration_s: float | None = None,
    error_summary: str | None = None,
) -> None:
    with engine.begin() as conn:
        result = conn.execute(
            update(model_runs)
            .where(model_runs.c.run_id == run_id)
            .values(
                status=STATUS_SUCCEEDED if succeeded else STATUS_FAILED,
                finished_at=_now(),
                primary_metric=primary_metric,
                primary_value=_clean_float(primary_value),
                n_rounds_final=n_rounds_final,
                n_rows=n_rows,
                n_features=n_features,
                duration_s=_clean_float(duration_s),
                error_summary=error_summary,
            )
        )
        if result.rowcount == 0:
            raise KeyError(f"unknown run_id {run_id}")


def mark_holdout_evaluated(engine: Engine, run_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(model_runs)
            .where(model_runs.c.run_id == run_id)
            .values(holdout_evaluated_at=_now())
        )


def get_run(engine: Engine, run_id: str) -> dict[str, Any] | None:
    ensure_tables(engine)
    with engine.connect() as conn:
        row = (
            conn.execute(select(model_runs).where(model_runs.c.run_id == run_id)).mappings().first()
        )
    return dict(row) if row else None


def list_runs(
    engine: Engine,
    *,
    experiment_name: str | None = None,
    experiment_run_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    ensure_tables(engine)
    stmt = select(model_runs)
    if experiment_name is not None:
        stmt = stmt.where(model_runs.c.experiment_name == experiment_name)
    if experiment_run_id is not None:
        stmt = stmt.where(model_runs.c.experiment_run_id == experiment_run_id)
    if status is not None:
        stmt = stmt.where(model_runs.c.status == status)
    stmt = stmt.order_by(model_runs.c.started_at.desc())
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings()]


def get_metrics(engine: Engine, run_id: str, split: str | None = None) -> list[dict[str, Any]]:
    ensure_tables(engine)
    stmt = select(run_metrics).where(run_metrics.c.run_id == run_id)
    if split is not None:
        stmt = stmt.where(run_metrics.c.split == split)
    stmt = stmt.order_by(run_metrics.c.split, run_metrics.c.fold, run_metrics.c.metric)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings()]


def get_feature_importance(
    engine: Engine, run_id: str, *, split: str | None = None, method: str | None = None
) -> list[dict[str, Any]]:
    ensure_tables(engine)
    stmt = select(run_feature_importance).where(run_feature_importance.c.run_id == run_id)
    if split is not None:
        stmt = stmt.where(run_feature_importance.c.split == split)
    if method is not None:
        stmt = stmt.where(run_feature_importance.c.method == method)
    stmt = stmt.order_by(
        run_feature_importance.c.method,
        run_feature_importance.c.fold,
        run_feature_importance.c.feature,
    )
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings()]


def get_params(engine: Engine, run_id: str) -> dict[str, str]:
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(select(run_params).where(run_params.c.run_id == run_id)).mappings()
        return {r["key"]: r["value"] for r in rows}
