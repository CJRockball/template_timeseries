"""Run one config or a sweep, recording everything in project.db and the run folder.

Order inside one run: write the ``running`` row, do the work, write the files,
write metrics, then set the final status. A ``succeeded`` row therefore means the
files exist. The holdout is not touched here; ``evaluation.evaluate_run`` does that.

Standard output of every run (all models, so runs stay comparable):
    run_metrics            standard metrics per fold, fold mean and std, base rate
    run_feature_importance permutation importance per fold, with spread across folds
    fold_models/           the model that predicted each validation fold
    error_sample.csv       most confident misses and false alarms with feature values
    model_runs             duration, rows, features, rounds, git state
An analysis that fails is logged and listed in ``analysis_errors.json``; it does not
fail the run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from ml_template.db import model_run_registry as registry
from ml_template.evaluation.error_sample import error_sample
from ml_template.evaluation.learning_curves import learning_curves_for_run
from ml_template.evaluation.permutation_importance import importance_rows_for_run
from ml_template.models.model_factory import AdapterFactory, build_adapter_factory
from ml_template.paths import ARTIFACTS_RUNS, DATA_SILVER, DATA_SPLITS, LOGS
from ml_template.tracking import run_artifacts
from ml_template.tracking.git_state import git_state
from ml_template.tracking.pipeline_logging import run_log
from ml_template.training.cross_validation import cross_validate, fit_final
from ml_template.training.run_config import RunConfig, config_hash, config_json, flatten_params
from ml_template.training.run_inputs import load_run_inputs

logger = logging.getLogger("ml_template.training.experiment_runner")

FactoryBuilder = Callable[[RunConfig], AdapterFactory]


def _run_analyses(
    cfg: RunConfig,
    engine: Engine,
    run_id: str,
    run_dir: Path,
    inputs: Any,
    cv: Any,
    make_adapter: AdapterFactory,
) -> None:
    """Standard analyses. Each one is isolated so a failure cannot lose the run."""
    key_col = inputs.contract.entity_id
    errors: dict[str, str] = {}

    if cfg.analysis.permutation_importance:
        try:
            rows = importance_rows_for_run(
                cv.fold_models,
                cv.fold_val_keys,
                inputs.X,
                inputs.y,
                n_repeats=cfg.analysis.permutation_repeats,
                seed=cfg.seed,
                max_rows=cfg.analysis.permutation_max_rows,
            )
            registry.log_feature_importance(engine, run_id, rows)
            pooled = sorted(
                (r for r in rows if r["fold"] == -1), key=lambda r: r["value"], reverse=True
            )
            logger.info(
                "Run %s: top permutation importance %s",
                run_id,
                ", ".join(f"{r['feature']}={r['value']:.4f}" for r in pooled[:3]),
            )
        except Exception as exc:
            logger.exception("Run %s: permutation importance failed", run_id)
            errors["permutation_importance"] = f"{type(exc).__name__}: {exc}"

    if cfg.analysis.error_sample_size > 0:
        try:
            sample = error_sample(
                cv.oof, inputs.X, inputs.y, key_col, cfg.analysis.error_sample_size
            )
            sample.to_csv(run_dir / "error_sample.csv", index=False)
        except Exception as exc:
            logger.exception("Run %s: error sample failed", run_id)
            errors["error_sample"] = f"{type(exc).__name__}: {exc}"

    if cfg.analysis.learning_curves:
        try:
            table = learning_curves_for_run(
                make_adapter,
                inputs.X,
                inputs.y,
                inputs.assignments,
                key_col,
                cv.folds,
                metric=cfg.analysis.learning_curve_metric,
                max_rounds=cfg.analysis.learning_curve_rounds,
            )
            table.to_parquet(run_dir / "learning_curves.parquet", index=False)
        except Exception as exc:
            logger.exception("Run %s: learning curves failed", run_id)
            errors["learning_curves"] = f"{type(exc).__name__}: {exc}"

    if errors:
        run_artifacts.write_json(run_dir / "analysis_errors.json", errors)


def _train_and_record(
    cfg: RunConfig,
    engine: Engine,
    run_id: str,
    run_dir: Path,
    *,
    silver_root: Path,
    split_root: Path,
    factory_builder: FactoryBuilder,
) -> dict[str, Any]:
    run_artifacts.write_json(run_dir / "environment.json", run_artifacts.environment_info())
    (run_dir / "config.json").write_text(config_json(cfg), encoding="utf-8")
    inputs = load_run_inputs(cfg, silver_root=silver_root, split_root=split_root)
    key_col = inputs.contract.entity_id
    logger.info(
        "Run %s: %d rows, %d features, split %s",
        run_id,
        len(inputs.X),
        inputs.X.shape[1],
        inputs.split_id,
    )
    make_adapter = factory_builder(cfg)
    cv = cross_validate(
        make_adapter,
        inputs.X,
        inputs.y,
        inputs.assignments,
        key_col,
        validation_fraction=cfg.training.validation_fraction,
        seed=cfg.seed,
        metric_names=cfg.metrics.names,
        alert_rates=cfg.metrics.alert_rates,
        refit_rounds_factor=cfg.training.refit_rounds_factor,
        **{"task": cfg.task} if cfg.task == "regression" else {},
    )
    final = fit_final(
        make_adapter, inputs.X, inputs.y, inputs.assignments, key_col, cv.n_rounds_final
    )
    final.save(run_dir / "model.ubj")
    fold_dir = run_dir / "fold_models"
    fold_dir.mkdir()
    for fold, adapter in cv.fold_models.items():
        adapter.save(fold_dir / f"fold_{fold}.ubj")
    cv.oof.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    cv.folds.to_csv(run_dir / "cv_folds.csv", index=False)
    from ml_template.evaluation.curve_history import save_inner_history

    save_inner_history(run_dir, cv)
    run_artifacts.write_json(
        run_dir / "features.json",
        {
            "features": list(inputs.X.columns),
            "dtypes": {c: str(t) for c, t in inputs.X.dtypes.items()},
            "categorical_schema": {
                c: {
                    "categories": inputs.X[c].cat.categories.tolist(),
                    "ordered": bool(inputs.X[c].cat.ordered),
                }
                for c in inputs.X
                if hasattr(inputs.X[c].dtype, "categories")
            },
            "fitted_features": cfg.fitted_features,
            "n_rounds_final": cv.n_rounds_final,
            "silver_dataset_version_id": cfg.data.silver_dataset_version_id,
            "split_id": inputs.split_id,
        },
    )
    registry.log_metrics(engine, run_id, cv.metric_rows)
    if cfg.task == "regression":
        run_artifacts.write_json(
            run_dir / "task.json",
            {
                "task": cfg.task,
                "prediction_column": "y_pred_oof",
                "primary_metric": cfg.metrics.primary,
                "metric_direction": "min",
                "unsupported": [
                    "evaluation",
                    "calibration",
                    "release",
                    "scoring",
                    "permutation_importance",
                    "error_samples",
                ],
            },
        )
        from ml_template.evaluation.curve_history import run_regression_curve_diagnostics

        run_regression_curve_diagnostics(cfg, run_dir, inputs, cv, make_adapter, logger)
    else:
        _run_analyses(cfg, engine, run_id, run_dir, inputs, cv, make_adapter)
    primary = next(
        r["value"]
        for r in cv.metric_rows
        if r["split"] == "cv" and r["fold"] == -1 and (r["metric"] == cfg.metrics.primary)
    )
    run_artifacts.write_artifact_manifest(run_dir)
    logger.info(
        "Run %s: cv %s=%.4f, final rounds=%d",
        run_id,
        cfg.metrics.primary,
        primary,
        cv.n_rounds_final,
    )
    return {
        "primary_value": primary,
        "n_rounds_final": cv.n_rounds_final,
        "n_rows": len(inputs.X),
        "n_features": inputs.X.shape[1],
    }


def execute_run(
    cfg: RunConfig,
    engine: Engine,
    *,
    experiment_run_id: str,
    run_root: Path = ARTIFACTS_RUNS,
    log_root: Path = LOGS / "runs",
    silver_root: Path = DATA_SILVER,
    split_root: Path = DATA_SPLITS,
    factory_builder: FactoryBuilder = build_adapter_factory,
) -> str:
    """Train one config. Returns the run_id. Re-raises after recording a failure."""
    from ml_template.data.contract import load_dataset_contract
    from ml_template.training.run_config import resolve_path
    from ml_template.training.run_inputs import split_id_for

    started = time.perf_counter()
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    run_id = registry.new_run_id("run")
    run_dir = run_artifacts.create_run_dir(run_id, run_root)
    git = git_state()
    registry.start_model_run(
        engine,
        run_id=run_id,
        experiment_run_id=experiment_run_id,
        experiment_name=cfg.experiment_name,
        config_hash=config_hash(cfg),
        params=flatten_params(cfg),
        silver_dataset_version_id=cfg.data.silver_dataset_version_id,
        feature_set_id=cfg.data.feature_set_id,
        split_id=split_id_for(cfg, contract),
        model_type=cfg.model.type,
        seed=cfg.seed,
        artifact_dir=run_artifacts.relative_to_root(run_dir),
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
    )

    with run_log(Path(log_root) / f"{run_id}.log"):
        try:
            outcome = _train_and_record(
                cfg,
                engine,
                run_id,
                run_dir,
                silver_root=silver_root,
                split_root=split_root,
                factory_builder=factory_builder,
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            registry.finish_model_run(
                engine,
                run_id,
                succeeded=False,
                duration_s=time.perf_counter() - started,
                error_summary=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise
        registry.finish_model_run(
            engine,
            run_id,
            succeeded=True,
            primary_metric=cfg.metrics.primary,
            primary_value=outcome["primary_value"],
            n_rounds_final=outcome["n_rounds_final"],
            n_rows=outcome["n_rows"],
            n_features=outcome["n_features"],
            duration_s=time.perf_counter() - started,
        )
    return run_id


def run_experiment(
    configs: list[RunConfig],
    engine: Engine,
    *,
    kind: str,
    sweep_path: str | None = None,
    fail_fast: bool = False,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run configs under one launch (one experiment_run_id)."""
    if not configs:
        raise ValueError("no configs to run")
    names = {c.experiment_name for c in configs}
    if len(names) != 1:
        raise ValueError(f"a launch needs one experiment_name, got {sorted(names)}")
    (name,) = names

    git = git_state()
    launch_id = registry.start_launch(
        engine,
        experiment_name=name,
        kind=kind,
        n_runs=len(configs),
        sweep_path=sweep_path,
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
    )
    logger.info("Launch %s: %d runs for experiment %s", launch_id, len(configs), name)

    run_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for cfg in configs:
        try:
            run_ids.append(execute_run(cfg, engine, experiment_run_id=launch_id, **run_kwargs))
        except Exception as exc:
            failures.append(
                {"config_hash": config_hash(cfg), "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            if fail_fast:
                break

    registry.finish_launch(
        engine,
        launch_id,
        succeeded=not failures,
        n_failed=len(failures),
        error_summary="; ".join(f["error"] for f in failures)[:500] or None,
    )
    logger.info(
        "Launch %s finished: %d succeeded, %d failed", launch_id, len(run_ids), len(failures)
    )
    return {"experiment_run_id": launch_id, "run_ids": run_ids, "failures": failures}
