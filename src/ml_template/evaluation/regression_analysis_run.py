"""Versioned diagnostics for saved regression folds; never load holdout targets."""

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_template.evaluation.regression_diagnostics import compute_diagnostics, write_diagnostics


def load_development(prepared):
    from ml_template.features.prepared_recipe import load_partition

    keys = prepared.assignments.loc[
        prepared.assignments.partition.eq("dev"), prepared.key_col
    ].tolist()
    return load_partition(prepared, keys)


def prepare_analysis(engine, run_id):
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_regression
    from ml_template.models.predictor_selection import describe_predictor
    from ml_template.tracking.run_artifacts import verify_artifact

    # Only metadata preparation is reused, not evaluation execution/registration.
    p = prepare_regression(engine, RegressionEvaluationConfig(parent_run_id=run_id))
    folds = sorted(
        int(k) for k in p.assignments.loc[p.assignments.partition.eq("dev"), "fold"].unique()
    )
    descriptor = describe_predictor(
        p.run_cfg, p.run_dir, "cv_mean", fold_ids=folds, verify_artifact=verify_artifact
    )
    return p, descriptor


def resolve_options(
    cfg,
    *,
    metric=None,
    repeats=None,
    max_rows=None,
    seed=None,
    error_sample_size=None,
    permutation=True,
    plots=True,
    target_unit=None,
):
    configured = getattr(cfg.analysis, "regression_permutation_metric", "primary")
    chosen = metric or (cfg.metrics.primary if configured == "primary" else configured)
    options = {
        "metric": chosen,
        "n_repeats": repeats if repeats is not None else cfg.analysis.permutation_repeats,
        "max_rows": max_rows if max_rows is not None else cfg.analysis.permutation_max_rows,
        "seed": cfg.seed if seed is None else seed,
        "error_sample_size": 25 if error_sample_size is None else error_sample_size,
        "permutation": permutation,
        "target_unit": target_unit
        or getattr(cfg.analysis, "regression_target_unit", "target_units"),
    }
    if options["metric"] not in {"rmse", "mae"}:
        raise ValueError("Regression importance metric must be rmse or mae")
    for name, minimum in [("n_repeats", 1), ("seed", 0), ("error_sample_size", 0)]:
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid integer option: {name}")
    cap = options["max_rows"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("Positive integer max_rows required")
    if not isinstance(options["target_unit"], str) or not options["target_unit"].strip():
        raise ValueError("Document a nonempty target unit or use target_units")
    return options, plots


def ensure_analysis_destination(directory, run_dir):
    directory = Path(directory).resolve()
    if directory.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("Standalone analyses must not be written inside the sealed training run")
    for parent in [directory, *directory.parents]:
        if (parent / "artifact_manifest.json").is_file() or (parent / "manifest.json").is_file():
            raise ValueError(
                "Standalone analyses must not be nested under sealed artifact directories"
            )


def run_saved_analysis(
    engine,
    run_id,
    *,
    check_only=True,
    output_root=None,
    metric=None,
    repeats=None,
    max_rows=None,
    seed=None,
    error_sample_size=None,
    permutation=True,
    plots=True,
    target_unit=None,
):
    from ml_template.evaluation.evaluation_artifacts import manifest, sha256, write_json
    from ml_template.models.model_factory import load_adapter
    from ml_template.models.predictor_selection import safe_file, verify_predictor_files
    from ml_template.scoring.release_artifacts import runtime_info

    p, descriptor = prepare_analysis(engine, run_id)
    options, plots = resolve_options(
        p.run_cfg,
        metric=metric,
        repeats=repeats,
        max_rows=max_rows,
        seed=seed,
        error_sample_size=error_sample_size,
        permutation=permutation,
        plots=plots,
        target_unit=target_unit,
    )
    summary = {
        "ready": True,
        "parent_run_id": run_id,
        "task": "regression",
        "n_development": int(p.assignments.partition.eq("dev").sum()),
        "options": options,
        "holdout_labels_loaded": False,
        "development_labels_loaded": False,
        "parent_artifacts_modified": False,
        "files_written": False,
    }
    if check_only:
        return summary
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "analyses"
    identity = f"analysis_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:10]}"
    directory = Path(output_root).resolve() / identity
    ensure_analysis_destination(directory, p.run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        **summary,
        "analysis_id": identity,
        "artifact_dir": str(directory),
        "status": "running",
    }
    write_json(directory / "analysis.json", state)
    try:
        X, y = load_development(p)
        oof = pd.read_parquet(
            p.run_dir / "oof_predictions.parquet", columns=[p.key_col, "fold", "y_pred_oof"]
        )
        names = {m["fold_id"]: m["model_path"] for m in descriptor["members"]}

        def loader(fold):
            member = next(m for m in descriptor["members"] if m["fold_id"] == fold)
            required = [names[fold]]
            if member["transformer_path"]:
                required.append(member["transformer_path"])
            verify_predictor_files(p.run_dir, descriptor, names=required)
            return load_adapter(p.run_cfg, safe_file(p.run_dir, names[fold]))

        result = compute_diagnostics(
            oof, X, y, p.assignments, p.key_col, model_loader=loader, **options
        )
        if sha256(p.parquet_path) != p.provenance["input_hashes"]["silver_parquet"]:
            raise ValueError("Silver bytes changed while analysing development rows")
        from ml_template.tracking.run_artifacts import verify_artifact

        for name in ["config.json", "features.json", "oof_predictions.parquet"]:
            verify_artifact(p.run_dir, name)
        verify_predictor_files(p.run_dir, descriptor)
        from ml_template.features.prepared_recipe import verify_prepared

        verify_prepared(p)
        write_diagnostics(result, directory, plots=plots)
        write_json(
            directory / "provenance.json",
            {
                "parent_run_id": run_id,
                "input_provenance": p.provenance,
                "fold_descriptor": descriptor,
                "runtime": runtime_info(),
                "options": options,
                "sample_keys": result.samples,
                "registry_rows_written": False,
                "fold_descriptor_usage": "Inventory integrity only: individual folds predict their own validation rows; no CV-mean prediction on development rows",
            },
        )
        state.update(
            status="succeeded",
            files_written=True,
            development_labels_loaded=True,
            report_path=str(directory / "report.html"),
        )
        write_json(directory / "analysis.json", state)
        manifest(directory)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "analysis.json", state)
        raise
    return state


def run_training_diagnostics(cfg, engine, run_id, run_dir, inputs, cv, logger):
    """Only called while the new training run is still open, before its seal."""
    if not cfg.analysis.permutation_importance and cfg.analysis.error_sample_size == 0:
        return
    from ml_template.db import model_run_registry as registry
    from ml_template.evaluation.evaluation_artifacts import write_json

    directory = Path(run_dir) / "regression_diagnostics"
    try:
        directory.mkdir(exist_ok=False)
        options, _ = resolve_options(
            cfg,
            error_sample_size=cfg.analysis.error_sample_size,
            permutation=cfg.analysis.permutation_importance,
        )
        result = compute_diagnostics(
            cv.oof,
            inputs.X,
            inputs.y,
            inputs.assignments,
            inputs.contract.entity_id,
            model_loader=lambda fold: cv.fold_models[fold],
            **options,
        )
        write_diagnostics(result, directory, plots=True)
        if result.registry_rows:
            registry.log_feature_importance(engine, run_id, result.registry_rows)
    except Exception as exc:
        logger.exception("Regression OOF diagnostics failed")
        path = Path(run_dir) / "analysis_errors.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        write_json(path, {**old, "regression_diagnostics": f"{type(exc).__name__}: {exc}"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metric", choices=["rmse", "mae"])
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--error-sample-size", type=int, default=25)
    parser.add_argument("--target-unit")
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from ml_template.db.connection import get_engine

    result = run_saved_analysis(
        get_engine(),
        args.run_id,
        check_only=args.check,
        output_root=args.output_root,
        metric=args.metric,
        repeats=args.repeats,
        max_rows=args.max_rows,
        seed=args.seed,
        error_sample_size=args.error_sample_size,
        permutation=not args.no_permutation,
        plots=not args.no_plots,
        target_unit=args.target_unit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
