"""Package frozen regression evidence; never train or read target columns."""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.scoring.release_artifacts import (
    inside,
    read_json,
    runtime_info,
    seal,
    sha256,
    verify,
    write_json,
)


def regression_evidence(engine, request):
    from ml_template.scoring.nn_release import evidence, is_nn_parent

    if is_nn_parent(engine, request.parent_run_id):
        return evidence(engine, request)
    from ml_template.db.evaluation_registry import get_evaluation
    from ml_template.evaluation.evaluation_inputs import digest
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_evaluate import _verify_cache
    from ml_template.evaluation.regression_inputs import prepare_regression
    from ml_template.tracking.run_artifacts import resolve_stored_path

    record = get_evaluation(engine, request.evaluation_id)
    if not record or record["status"] != "succeeded":
        raise ValueError("An explicit successful regression evaluation is required")
    _verify_cache(record, record["identity_hash"])
    directory = resolve_stored_path(record["artifact_dir"])
    spec = RegressionEvaluationConfig.model_validate(read_json(directory / "spec.json"))
    registered = RegressionEvaluationConfig.model_validate_json(record["spec_json"])
    if spec != registered or spec.parent_run_id != request.parent_run_id:
        raise ValueError("Release request/evaluation registry/spec mismatch")
    summary = read_json(directory / "summary.json")
    provenance = read_json(directory / "provenance.json")
    if (
        summary["evaluation_id"] != request.evaluation_id
        or summary["parent_run_id"] != request.parent_run_id
        or summary.get("task") != "regression"
    ):
        raise ValueError("Evaluation summary binding differs")
    if (
        summary["identity_hash"] != record["identity_hash"]
        or digest({"spec": spec.model_dump(mode="json"), "provenance": provenance})
        != record["identity_hash"]
    ):
        raise ValueError("Evaluation evidence digest differs")
    if hasattr(request, "calibration_id") and request.calibration_id != spec.calibration_id:
        raise ValueError("release request calibration differs from evaluation")
    p = prepare_regression(engine, spec)
    if (
        p.spec != spec
        or p.provenance["predictor"] != provenance["predictor"]
        or p.provenance["input_hashes"] != provenance["input_hashes"]
    ):
        raise ValueError("Evaluated predictor/data artifacts changed")
    current_code = runtime_info()["code_sha256"]
    from ml_template.features.prepared_recipe import verify_prediction_code

    verify_prediction_code(provenance["code_hashes"], current_code)
    return (p, directory, spec, summary, provenance)


def prepare_regression_release(engine, request):
    from ml_template.features.feature_set_store import resolve_feature_set
    from ml_template.features.prepared_recipe import prepare_frame, verify_prepared
    from ml_template.scoring.input_contract import coerce_frame, validate_batch
    from ml_template.scoring.nn_release import neural_config
    from ml_template.scoring.pre_gold_schema import build_pre_gold_contract
    from ml_template.scoring.source_spec import freeze_source_spec

    p, directory, spec, summary, provenance = regression_evidence(engine, request)
    plan = provenance["feature_preparation"]
    raw_columns = plan["recipe"]["raw_columns"]
    # Features only: never request target or y_true for packaging or parity.
    source = pd.read_parquet(p.parquet_path, columns=[p.key_col, *raw_columns])
    if (
        source[p.key_col].isna().any()
        or source[p.key_col].duplicated().any()
        or set(source[p.key_col]) != set(p.assignments[p.key_col])
    ):
        raise ValueError("Source key coverage differs")
    source = source.set_index(p.key_col).sort_index()
    model_features = prepare_frame(source, plan["recipe"], plan["model_schema"])
    cached = source
    if p.run_cfg.data.feature_set_id:
        cached = resolve_feature_set(
            p.run_cfg.data.feature_set_id,
            source,
            p.run_cfg.data.silver_dataset_version_id,
            p.roots["feature_root"],
        )
    cached = coerce_frame(cached, plan["model_schema"])
    pd.testing.assert_frame_equal(cached, model_features)
    if sha256(p.parquet_path) != provenance["input_hashes"]["silver_parquet"]:
        raise ValueError("Silver changed while projecting raw features")
    verify_prepared(p)
    input_spec, source_policy, model_policy = freeze_source_spec(p, source)
    keys = source.index.to_series()
    is_int = keys.map(
        lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
    ).all()
    is_str = keys.map(lambda v: isinstance(v, str)).all()
    if not is_int and not is_str:
        raise ValueError("Entity keys must be homogeneous integer or string")
    contract = build_pre_gold_contract(
        source,
        model_features,
        key_col=p.key_col,
        key_kind="int" if is_int else "str",
        input_spec=input_spec,
        source_policy=source_policy,
        model_policy=model_policy,
    )
    if neural_config(getattr(p, "run_cfg", None)):
        contract["estimator_validation"] = "saved_nn_adapter_finite_arrays"
    validate_batch(source.reset_index(), contract, enforce_rates=True)
    return {
        "prepared": p,
        "evaluation_dir": directory,
        "spec": spec,
        "summary": summary,
        "provenance": provenance,
        "source": source,
        "contract": contract,
        "recipe": plan["recipe"],
        "reference_sha256": read_json(directory / "artifact_manifest.json")["files"][
            "holdout_predictions.parquet"
        ]["sha256"],
    }


def check_regression_parity(predictor, material, atol):
    from ml_template.scoring.nn_release import check_parity, neural_config

    if neural_config(getattr(material["prepared"], "run_cfg", None)):
        return check_parity(predictor, material, atol)
    p = material["prepared"]
    path = material["evaluation_dir"] / "holdout_predictions.parquet"
    expected_hash = material["reference_sha256"]
    if sha256(path) != expected_hash:
        raise ValueError("Evaluation predictions changed")
    from ml_template.evaluation.regression_postprocessing import reference_columns

    extra = reference_columns(material)
    reference = pd.read_parquet(path, columns=[p.key_col, "y_pred", *extra])
    if (
        reference[p.key_col].isna().any()
        or reference[p.key_col].duplicated().any()
        or set(reference[p.key_col]) != set(p.hold_keys)
    ):
        raise ValueError("Evaluation prediction keys differ")
    reference = reference.set_index(p.key_col).sort_index()
    values = reference.y_pred.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation predictions")
    batch = material["source"].loc[reference.index].reset_index()
    whole = predictor.predict(batch).set_index(p.key_col).sort_index()
    chunked = (
        predictor.predict(
            batch.iloc[::-1].reset_index(drop=True), chunk_size=max(1, min(137, len(batch) // 3))
        )
        .set_index(p.key_col)
        .sort_index()
    )
    for column in [
        "y_pred_raw",
        "y_pred_release",
        *[name for name in extra if name != "y_pred_raw"],
    ]:
        expected_values = reference[column].to_numpy(float) if column in reference else values
        if not np.allclose(expected_values, whole[column], rtol=0, atol=atol) or not np.allclose(
            whole[column], chunked[column], rtol=0, atol=atol
        ):
            raise ValueError(f"Regression prediction parity failed: {column}")
    singleton = predictor.predict(batch.iloc[:1]).y_pred_release.iloc[0]
    if abs(singleton - whole.y_pred_release.iloc[0]) > atol:
        raise ValueError("Regression singleton parity failed")
    if sha256(path) != expected_hash:
        raise ValueError("Evaluation predictions changed during parity")
    return {
        "n_rows": len(reference),
        "atol": atol,
        "max_abs_diff": float(np.max(np.abs(values - whole.y_pred_release.to_numpy()))),
        "keyed": True,
        "labels_loaded": False,
        "metrics_recomputed": False,
        "shuffled_chunk_parity": True,
        "singleton_parity": True,
    }


def build_regression_release(engine, request, *, check_only=False, output_root=None):
    from ml_template.scoring.nn_release import neural_config

    material = prepare_regression_release(engine, request)
    p, spec = (material["prepared"], material["spec"])
    if check_only:
        if neural_config(getattr(p, "run_cfg", None)):
            from ml_template.paths import ARTIFACTS
            from ml_template.scoring.nn_release import validate_destination

            validate_destination(p, output_root or ARTIFACTS / "releases")
        return {
            "ready": True,
            "task": "regression",
            "parent_run_id": request.parent_run_id,
            "evaluation_id": request.evaluation_id,
            "predictor_kind": spec.predictor.kind,
            "holdout_labels_loaded": False,
            "parity_executed": False,
            "files_written": False,
            **(
                {
                    "model_type": "nn",
                    "models_loaded": False,
                    "predictions_computed": False,
                    "source_features_loaded": True,
                    "labels_loaded": False,
                }
                if neural_config(getattr(p, "run_cfg", None))
                else {}
            ),
        }
    from ml_template.db.scoring_registry import register_release
    from ml_template.paths import ARTIFACTS
    from ml_template.scoring.predictor import ReleasedPredictor

    root = Path(output_root or ARTIFACTS / "releases").resolve()
    if neural_config(getattr(p, "run_cfg", None)):
        from ml_template.scoring.nn_release import validate_destination

        validate_destination(p, root)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".regression-release-", dir=root))
    try:
        policy = {
            "kind": "none",
            "value": None,
            "source": "Regression has no classification policy",
            "batch_definition": "entire scoring input",
            "selection": None,
        }
        info = {
            "schema_version": 2,
            "task": "regression",
            "prediction_kind": "continuous",
            "name": request.name,
            "created_at": datetime.now(UTC).isoformat(),
            "parent_run_id": request.parent_run_id,
            "evaluation_id": request.evaluation_id,
            "calibration_id": spec.calibration_id,
            "release_variant": spec.release_variant,
            "policy": policy,
            "intended_use": request.intended_use,
            "research_only": True,
            "automatic_promotion": False,
            "predictor_kind": spec.predictor.kind,
            "prediction_columns": {"raw": "y_pred_raw", "release": "y_pred_release"},
            "evaluation_summary": material["summary"],
            "limitations": list(spec.limitations),
        }
        if neural_config(getattr(p, "run_cfg", None)):
            info.update(model_type="nn", target_unit=spec.target_unit)
        from ml_template.evaluation.regression_postprocessing import package

        package(stage, material, info)
        write_json(stage / "release.json", info)
        write_json(stage / "contract.json", material["contract"])
        write_json(stage / "feature_recipe.json", material["recipe"])
        write_json(stage / "runtime.json", runtime_info())
        descriptor = material["provenance"]["predictor"]
        write_json(stage / "predictor.json", descriptor)
        write_json(
            stage / "evidence.json",
            {
                "provenance": material["provenance"],
                "evaluation_spec": spec.model_dump(mode="json"),
                "evaluation_summary": material["summary"],
            },
        )
        for name in dict.fromkeys(["config.json", "features.json", *descriptor["files_sha256"]]):
            dest = inside(stage, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.run_dir / name, dest)
            if name in {"config.json", "features.json"}:
                expected = material["provenance"]["input_hashes"][name.removesuffix(".json")]
            else:
                expected = descriptor["files_sha256"][name]
            if sha256(dest) != expected:
                raise ValueError("Copied predictor artifact changed: " + name)
        seal(stage)
        parity = check_regression_parity(ReleasedPredictor(stage), material, request.parity_atol)
        write_json(stage / "parity.json", parity)
        manifest = seal(stage)
        verify(stage)
        destination = root / manifest["release_id"]
        if destination.exists():
            raise FileExistsError("Immutable release destination exists")
        stage.rename(destination)
        register_release(engine, manifest, info, destination)
        return {
            "release_id": manifest["release_id"],
            "artifact_dir": str(destination),
            "task": "regression",
            "status": "succeeded",
            "parity": parity,
            "holdout_labels_loaded": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
