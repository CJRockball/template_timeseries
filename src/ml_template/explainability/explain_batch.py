"""Explain a complete saved scoring batch through its exact pinned release runtime."""

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.explainability.artifacts import (
    read_json,
    seal_explanation,
    sha256,
    verify_explanation,
    write_json,
)
from ml_template.explainability.explanation_config import load_config
from ml_template.explainability.explanation_inputs import inspect_saved_batch, load_saved_batch
from ml_template.explainability.runtime_bridge import run_worker


def check_scoring_record(config, saved):
    database = Path(config.registry_path)
    if not database.is_file():
        raise ValueError("existing project SQLite database required")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        record = conn.execute(
            "SELECT release_id,status,input_sha256,n_rows,n_flagged FROM scoring_batches WHERE scoring_batch_id=?",
            (saved.summary["scoring_batch_id"],),
        ).fetchone()
    expected = (
        saved.summary["release_id"],
        "succeeded",
        saved.summary["input_sha256"],
        saved.summary["n_rows"],
        saved.summary["n_flagged"],
    )
    if record != expected:
        raise ValueError("scoring registry is missing, unfinished or differs from saved batch")


def validate_groups(config, names):
    if not config.groups:
        return
    assigned = []
    for name, columns in config.groups.items():
        if not name or not columns or not all(isinstance(c, str) and c for c in columns):
            raise ValueError("nonempty group names and feature lists required")
        assigned.extend(columns)
    if len(set(assigned)) != len(assigned) or set(assigned) != set(names):
        raise ValueError("groups must partition all estimator features exactly once")


def authoring_runtime():
    import ml_template

    package = Path(ml_template.__file__).resolve().parent
    return {
        "code_sha256": {
            p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*.py"))
        },
        "purpose": "explanation_orchestration_not_released_inference",
    }


def preflight(config):
    if getattr(config, "task", "binary_classification") == "regression":
        from ml_template.explainability.regression_batch import explain_batch as explain_regression

        return explain_regression(config, check_only=True)
    saved = load_saved_batch(config)
    check_scoring_record(config, saved)
    with tempfile.TemporaryDirectory(prefix="ml-explain-check-") as temporary:
        response = run_worker(config, saved.binding, temporary, check_only=True)
    return {
        "ready": True,
        "release_id": config.expected_release_id,
        "scoring_batch_id": config.expected_scoring_batch_id,
        "n_rows": len(saved.frame),
        "runtime_sha256": response["runtime_sha256"],
        "predictions_computed": False,
        "project_artifacts_written": False,
        "registry_written": False,
        "transient_ipc_files_used": True,
        "labels_loaded": False,
        "groups_checked_against_estimator": False,
    }


def read_worker_result(config, saved, response, directory):
    from ml_template.explainability.result import ExplanationTolerance, build_margin_explanation

    semantics = {
        "method": "xgboost_native_tree_shap",
        "feature_dependence": "tree_path_dependent",
        "output_scale": "base_model_log_odds",
    }
    if any(response.get(name) != value for name, value in semantics.items()):
        raise ValueError("worker explanation semantics differ")
    if response["n_rows"] != len(saved.frame):
        raise ValueError("worker row count differs")
    features = pd.read_parquet(Path(directory) / "features.parquet")
    expected_index = pd.Index(
        saved.frame[saved.contract["key_col"]].to_numpy(), name=saved.contract["key_col"]
    )
    if (
        not features.index.equals(expected_index)
        or list(features.columns) != response["feature_names"]
    ):
        raise ValueError("worker feature coverage/order differs")
    if len(features.columns) != response["n_features"]:
        raise ValueError("worker feature count differs")
    validate_groups(config, list(features.columns))
    tolerance = ExplanationTolerance(
        **{
            name: getattr(config, name)
            for name in ("margin_atol", "margin_rtol", "probability_atol", "probability_rtol")
        }
    )
    with np.load(Path(directory) / "contributions.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"contributions", "base_values", "margins", "p_raw"}:
            raise ValueError("unexpected worker array schema")
        result = build_margin_explanation(
            features,
            arrays["contributions"],
            arrays["base_values"],
            arrays["margins"],
            arrays["p_raw"],
            tolerance=tolerance,
        )
    reproduced = pd.read_parquet(Path(directory) / "reproduced_predictions.parquet")
    if not reproduced.index.equals(expected_index):
        raise ValueError("reproduced prediction coverage/order differs")
    expected_columns = set(saved.predictions.columns) - {"scoring_batch_id"}
    if set(reproduced.columns) != expected_columns:
        raise ValueError("reproduced prediction schema differs")
    if not (reproduced.release_id == config.expected_release_id).all():
        raise ValueError("reproduced prediction release identity differs")
    for name in ("p_raw", "p_calibrated", "p_release"):
        if name not in saved.predictions:
            continue
        values = reproduced[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("reproduced probability invalid")
        if not np.allclose(
            values,
            saved.predictions[name],
            atol=config.probability_atol,
            rtol=config.probability_rtol,
        ):
            raise ValueError(f"saved probability parity failed: {name}")
    if not np.allclose(
        result.rows.p_raw,
        reproduced.p_raw,
        atol=config.probability_atol,
        rtol=config.probability_rtol,
    ):
        raise ValueError("contribution probability differs from reproduced scorer")
    if "policy_flag" in reproduced:
        if (
            not pd.api.types.is_bool_dtype(reproduced.policy_flag.dtype)
            or reproduced.policy_flag.isna().any()
        ):
            raise ValueError("reproduced policy flags invalid")
        if not np.array_equal(reproduced.policy_flag, saved.predictions.policy_flag):
            raise ValueError("full-batch policy parity failed")
    return result, reproduced


def feature_metadata(config, response):
    recipe = read_json(Path(config.release_dir) / "feature_recipe.json")
    raw = set(recipe["raw_columns"])
    block_by_column = {name: item for item in recipe["blocks"] for name in item["columns"]}
    fitted = response.get("fitted_feature_metadata", {})
    output = []
    for name in response["feature_names"]:
        if name in fitted:
            item = {"name": name, **fitted[name]}
        elif name in raw:
            item = {"name": name, "kind": "raw_input"}
        elif name in block_by_column:
            block = block_by_column[name]
            item = {
                "name": name,
                "kind": "stateless_feature",
                "block_id": block["block_id"],
                "block_name": block["block_name"],
                "derived_from": block["derived_from"],
                "dependency_granularity": "block_level",
            }
        else:
            raise ValueError(f"unresolved estimator feature provenance: {name}")
        output.append(item)
    return {"schema_version": 1, "features": output}


def write_outputs(stage, config, saved, response, result, reproduced, explanation_id, elapsed):
    from ml_template.explainability.report import write_report
    from ml_template.explainability.summaries import feature_summary, grouped_summary

    rows = result.rows.copy()
    for name in saved.predictions.columns:
        if name in rows:
            rows[f"{name}_saved"] = saved.predictions[name]
        else:
            rows[name] = saved.predictions[name]
    for name in ("p_calibrated", "p_release"):
        if name in reproduced:
            rows[f"{name}_reproduced"] = reproduced[name]
    rows["explanation_id"] = explanation_id
    rows.to_parquet(stage / "rows.parquet", index=True)
    result.features.to_parquet(stage / "features.parquet", index=True)
    result.contributions.to_parquet(stage / "contributions.parquet", index=True)
    ranking = feature_summary(result)
    ranking.to_parquet(stage / "global_summary.parquet", index=False)
    if config.groups:
        grouped_summary(result, config.groups).to_parquet(
            stage / "group_summary.parquet", index=False
        )
    summary = {
        "schema_version": 1,
        "artifact_state": "complete",
        "explanation_id": explanation_id,
        "release_id": config.expected_release_id,
        "scoring_batch_id": config.expected_scoring_batch_id,
        "parent_run_id": saved.release_info["parent_run_id"],
        "calibration_id": saved.release_info["calibration_id"],
        "n_rows": len(result.rows),
        "n_features": len(result.features.columns),
        "n_flagged": saved.summary["n_flagged"],
        "method": result.method,
        "output_scale": result.output_scale,
        "feature_dependence": result.feature_dependence,
        "persisted_scoring_batch_verified": True,
        "calibrated_probability_attribution": False,
        "policy_attribution": False,
        "population": "complete_original_scoring_batch",
        "duration_s": elapsed,
        "labels_loaded": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(stage / "config.json", config.model_dump())
    write_json(stage / "summary.json", summary)
    write_json(stage / "feature_metadata.json", feature_metadata(config, response))
    write_json(stage / "verification.json", {**result.verification, "saved_batch_parity": response})
    write_json(
        stage / "provenance.json",
        {
            "schema_version": 1,
            "source_binding": saved.binding,
            "binding_sha256": saved.binding["binding_sha256"],
            "inference_worker": response,
            "orchestration_runtime": authoring_runtime(),
            "compatibility_strategy": "isolated_exact_pinned_runtime_no_migration",
            "source_manifests_modified": False,
            "training_dashboard_modified": False,
            "future_dashboard": "separate_read_only_usage_application",
            "hashes_are_signatures": False,
        },
    )
    write_report(stage, summary, ranking)
    return summary


def explain_batch(config, *, check_only=False):
    if getattr(config, "task", "binary_classification") == "regression":
        from ml_template.explainability.regression_batch import explain_batch as explain_regression

        return explain_regression(config, check_only=check_only)
    if check_only:
        return preflight(config)
    # Read-only validation precedes all execution registration and model loading.
    saved = load_saved_batch(config)
    check_scoring_record(config, saved)
    orchestration_before = authoring_runtime()
    from ml_template.db import explanation_registry as registry

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    explanation_id = f"explain_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    destination = root / explanation_id
    if destination.exists():
        raise FileExistsError("immutable explanation destination exists")
    stage = Path(tempfile.mkdtemp(prefix=".explanation-staging-", dir=root))
    registered = False
    published = False
    started = time.perf_counter()
    try:
        registry.start(
            config.registry_path, explanation_id, config.model_dump(), saved.binding, destination
        )
        registered = True
        (stage / "execution.log").write_text(
            f"{datetime.now(UTC).isoformat()} started {explanation_id}\n"
        )
        with tempfile.TemporaryDirectory(prefix="ml-explain-worker-") as temporary:
            response = run_worker(config, saved.binding, temporary, check_only=False)
            result, reproduced = read_worker_result(config, saved, response, temporary)
        if inspect_saved_batch(config)[2] != saved.binding:
            raise ValueError("source inputs changed before publication")
        check_scoring_record(config, saved)
        summary = write_outputs(
            stage,
            config,
            saved,
            response,
            result,
            reproduced,
            explanation_id,
            time.perf_counter() - started,
        )
        with (stage / "execution.log").open("a") as log:
            log.write(f"{datetime.now(UTC).isoformat()} all parity checks passed\n")
        if inspect_saved_batch(config)[2] != saved.binding:
            raise ValueError("source inputs changed during output generation")
        if authoring_runtime() != orchestration_before:
            raise ValueError("orchestration source changed during explanation execution")
        manifest = seal_explanation(stage, explanation_id)
        verify_explanation(stage, explanation_id)
        stage.rename(destination)
        published = True
        registry.finish(
            config.registry_path,
            explanation_id,
            duration_s=time.perf_counter() - started,
            manifest=manifest,
            n_rows=summary["n_rows"],
            n_features=summary["n_features"],
        )
        return {
            "explanation_id": explanation_id,
            "status": "succeeded",
            "artifact_dir": str(destination),
            "release_id": config.expected_release_id,
            "scoring_batch_id": config.expected_scoring_batch_id,
            "n_rows": summary["n_rows"],
            "n_features": summary["n_features"],
        }
    except Exception as exc:
        failure_dir = destination if published else stage
        if not published and stage.exists():
            failed_root = root / "_failed"
            failed_root.mkdir(exist_ok=True)
            failure_dir = failed_root / explanation_id
            write_json(
                stage / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "complete": False}
            )
            stage.rename(failure_dir)
        if registered:
            try:
                registry.finish(
                    config.registry_path,
                    explanation_id,
                    duration_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                    artifact_dir=failure_dir,
                )
            except Exception as registry_error:
                exc.add_note(f"Also failed to finalize registry: {registry_error}")
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.project_root is None:
        from ml_template.paths import ROOT

        project_root = ROOT
    else:
        project_root = args.project_root
    config = load_config(args.config, project_root=project_root)
    result = explain_batch(config, check_only=args.check)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
