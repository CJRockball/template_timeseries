"""Assemble a research release from a completed evaluation; never train or refit."""

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.scoring.input_contract import validate_batch
from ml_template.scoring.pre_gold_schema import build_pre_gold_contract
from ml_template.scoring.release_artifacts import (
    inside,
    read_json,
    runtime_info,
    seal,
    sha256,
    verify,
    write_json,
)
from ml_template.scoring.release_config import load_release_config
from ml_template.scoring.source_spec import freeze_source_spec


def _evidence(engine, request):
    from ml_template.db.evaluation_registry import get_evaluation
    from ml_template.evaluation.evaluation_config import EvaluationConfig
    from ml_template.evaluation.evaluation_inputs import prepare_evaluation
    from ml_template.tracking.run_artifacts import resolve_stored_path

    record = get_evaluation(engine, request.evaluation_id)
    if not record or record["status"] != "succeeded":
        raise ValueError("an explicit successful final evaluation is required")
    directory = resolve_stored_path(record["artifact_dir"])
    manifest = read_json(directory / "artifact_manifest.json")["files"]
    required = {"spec.json", "summary.json", "provenance.json", "holdout_predictions.parquet"}
    if not required <= manifest.keys():
        raise ValueError("evaluation evidence is incomplete")
    for name, entry in manifest.items():
        if sha256(inside(directory, name)) != entry["sha256"]:
            raise ValueError(f"evaluation artifact changed: {name}")
    spec = EvaluationConfig.model_validate(read_json(directory / "spec.json"))
    saved_spec = EvaluationConfig.model_validate_json(record["spec_json"])
    if spec.model_dump(mode="json") != saved_spec.model_dump(mode="json"):
        raise ValueError("evaluation registry/spec mismatch")
    summary, old = read_json(directory / "summary.json"), read_json(directory / "provenance.json")
    if (
        summary["evaluation_id"] != request.evaluation_id
        or summary["identity_hash"] != record["identity_hash"]
        or summary["parent_run_id"] != spec.parent_run_id
        or summary["calibration_id"] != spec.calibration_id
        or summary["release_variant"] != spec.release_variant
    ):
        raise ValueError("evaluation summary identity mismatch")
    if (request.parent_run_id, request.calibration_id) != (spec.parent_run_id, spec.calibration_id):
        raise ValueError("release request differs from evaluated predictor")
    prepared = prepare_evaluation(engine, spec)
    current = prepared.provenance
    for key in [
        "parent_run_id",
        "silver_dataset_version_id",
        "feature_set_id",
        "split_id",
        "parent_files_sha256",
        "silver_parquet_sha256",
        "silver_manifest_sha256",
        "split_manifest_sha256",
        "split_assignments_sha256",
    ]:
        if old.get(key) != current.get(key):
            raise ValueError(f"predictor/data changed after evaluation: {key}")
    for name, expected in old.get("input_hashes", {}).items():
        if current["input_hashes"].get(name) != expected:
            raise ValueError(f"evaluated input artifact changed: {name}")
    now_code = runtime_info()["code_sha256"]
    for name, expected in old.get("code_hashes", {}).items():
        if (
            name.startswith(("features/", "models/", "preprocessing/"))
            or name == "training/run_config.py"
        ) and now_code.get(name) != expected:
            raise ValueError(f"evaluated prediction implementation changed: {name}")
    evaluated_predictor = old.get("predictor")
    current_predictor = current.get("predictor")
    if evaluated_predictor is not None and evaluated_predictor != current_predictor:
        raise ValueError("selected predictor artifacts changed after evaluation")
    if spec.predictor.kind == "cv_mean" and evaluated_predictor is None:
        raise ValueError("cv_mean requires an evaluation bound to the fold-average predictor")
    return prepared, directory, spec, summary, old


def _source_frames(prepared):
    from ml_template.features.feature_set_store import resolve_feature_set
    from ml_template.models.model_factory import prepare_features

    cfg = prepared.run_cfg
    directory = (
        prepared.roots["silver_root"] / cfg.data.dataset_name / cfg.data.silver_dataset_version_id
    )
    meta = read_json(directory / "manifest.json")
    roles = meta["roles"]
    columns = roles.get("feature", roles.get("features", []))
    if not columns:
        raise ValueError("Silver feature role missing")
    # Projection explicitly excludes target and outcome fields.
    source = pd.read_parquet(directory / "data.parquet", columns=[prepared.key_col, *columns])
    if source[prepared.key_col].isna().any() or source[prepared.key_col].duplicated().any():
        raise ValueError("invalid source keys")
    source = source.set_index(prepared.key_col).sort_index()
    features = source
    if cfg.data.feature_set_id:
        features = resolve_feature_set(
            cfg.data.feature_set_id,
            source,
            cfg.data.silver_dataset_version_id,
            prepared.roots["feature_root"],
        )
    return source, prepare_features(features)


def prepare_release(engine, request):
    if getattr(request, "task", None) == "regression":
        from ml_template.scoring.regression_release import prepare_regression_release

        return prepare_regression_release(engine, request)
    from ml_template.features.feature_recipe import export_recipe, replay_recipe
    from ml_template.models.model_factory import prepare_features

    prepared, directory, spec, summary, provenance = _evidence(engine, request)
    source, model_features = _source_frames(prepared)
    input_spec, source_policy, model_policy = freeze_source_spec(prepared, source)
    recipe = export_recipe(
        prepared.run_cfg.data.feature_set_id,
        list(source.columns),
        prepared.run_cfg.data.silver_dataset_version_id,
        prepared.roots["feature_root"],
    )
    pd.testing.assert_frame_equal(prepare_features(replay_recipe(source, recipe)), model_features)
    recorded = read_json(prepared.run_dir / "features.json").get("categorical_schema")
    current = {
        c: {
            "categories": model_features[c].cat.categories.tolist(),
            "ordered": bool(model_features[c].cat.ordered),
        }
        for c in model_features
        if isinstance(model_features[c].dtype, pd.CategoricalDtype)
    }
    if recorded is not None and recorded != current:
        raise ValueError("recorded training categorical vocabulary changed")
    if list(model_features.columns) != prepared.feature_names:
        raise ValueError("saved model feature order changed")
    key_values = source.index.to_series()
    is_int = key_values.map(
        lambda v: isinstance(v, (int, np.integer)) and (not isinstance(v, (bool, np.bool_)))
    ).all()
    is_str = key_values.map(lambda v: isinstance(v, str)).all()
    if not is_int and (not is_str):
        raise ValueError("only homogeneous integer or string entity keys supported")
    contract = build_pre_gold_contract(
        source,
        model_features,
        key_col=prepared.key_col,
        key_kind="int" if is_int else "str",
        input_spec=input_spec,
        source_policy=source_policy,
        model_policy=model_policy,
        legacy_model_limits=getattr(request, "model_max_missing_rate", {}),
    )
    validate_batch(source.reset_index(), contract, enforce_rates=True)
    return {
        "prepared": prepared,
        "evaluation_dir": directory,
        "spec": spec,
        "summary": summary,
        "provenance": provenance,
        "source": source,
        "contract": contract,
        "recipe": recipe,
    }


def _check_parity(predictor, material, atol):
    prepared, spec = material["prepared"], material["spec"]
    key = prepared.key_col
    columns = [key, "p_raw"]
    if spec.calibration_id:
        columns.append("p_calibrated")
    if spec.policy.kind != "none":
        columns.append("policy_flag")
    # Labels are not necessary: never request y_true.
    reference = pd.read_parquet(
        material["evaluation_dir"] / "holdout_predictions.parquet", columns=columns
    )
    if (
        reference[key].isna().any()
        or reference[key].duplicated().any()
        or set(reference[key]) != set(prepared.hold_keys)
    ):
        raise ValueError("reference prediction key coverage differs")
    reference = reference.set_index(key).sort_index()
    batch = material["source"].loc[reference.index].reset_index()
    got = predictor.predict(batch).set_index(key).sort_index()
    chunked = (
        predictor.predict(batch.iloc[::-1].reset_index(drop=True), chunk_size=137)
        .set_index(key)
        .sort_index()
    )
    errors = {}
    for column in columns[1:]:
        if column == "policy_flag":
            if (
                reference[column].isna().any()
                or not np.array_equal(reference[column], got[column])
                or not np.array_equal(got[column], chunked[column])
            ):
                raise ValueError("released policy differs from evaluation/shuffled prediction")
        else:
            values = reference[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("invalid reference probabilities")
            diff = float(np.max(np.abs(values - got[column].to_numpy())))
            errors[column] = diff
            if diff > atol or not np.allclose(got[column], chunked[column], rtol=0, atol=atol):
                chunk_diff = float(
                    np.max(
                        np.abs(
                            got[column].to_numpy(dtype=float)
                            - chunked[column].to_numpy(dtype=float)
                        )
                    )
                )
                raise ValueError(
                    f"prediction parity failed: {column}; "
                    f"reference_max_abs={diff:.9g}; "
                    f"chunk_shuffle_max_abs={chunk_diff:.9g}; "
                    f"atol={atol:.9g}"
                )
    whole = predictor.predict(batch, include_policy=False)
    single = predictor.predict(batch.iloc[:1], include_policy=False)
    if abs(single.p_release.iloc[0] - whole.p_release.iloc[0]) > atol:
        raise ValueError("singleton vocabulary/prediction parity failed")
    return {
        "n_rows": len(reference),
        "atol": atol,
        "max_abs_diff": errors,
        "keyed": True,
        "labels_loaded": False,
        "shuffled_chunk_parity": True,
        "singleton_parity": True,
        "policy_parity": spec.policy.kind != "none",
    }


def build_release(engine, request, *, check_only=False, output_root=None):
    if getattr(request, "task", None) == "regression":
        from ml_template.scoring.regression_release import build_regression_release

        return build_regression_release(
            engine, request, check_only=check_only, output_root=output_root
        )
    material = prepare_release(engine, request)
    spec, prepared = (material["spec"], material["prepared"])
    if check_only:
        return {
            "ready": True,
            "evaluation_id": request.evaluation_id,
            "parent_run_id": spec.parent_run_id,
            "release_variant": spec.release_variant,
            "policy": spec.policy.model_dump(mode="json"),
            "holdout_labels_loaded": False,
            "parity_executed": False,
            "files_written": False,
        }
    from ml_template.db.scoring_registry import register_release
    from ml_template.paths import ARTIFACTS, ROOT
    from ml_template.scoring.predictor import ReleasedPredictor

    root = Path(output_root or ARTIFACTS / "releases").resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".release-staging-", dir=root))
    try:
        selection_source_hashes = None
        if spec.policy.selection:
            path = Path(spec.policy.selection.artifact_path)
            original = path if path.is_absolute() else ROOT / path
            if sha256(original) != spec.policy.selection.artifact_sha256:
                raise ValueError("selection changed during package creation")
            shutil.copyfile(original, stage / "policy_selection.json")
            selection_source_hashes = read_json(original)["source_hashes"]
        info = {
            "schema_version": 1,
            "name": request.name,
            "created_at": datetime.now(UTC).isoformat(),
            "parent_run_id": spec.parent_run_id,
            "calibration_id": spec.calibration_id,
            "evaluation_id": request.evaluation_id,
            "release_variant": spec.release_variant,
            "intended_use": request.intended_use,
            "policy": spec.policy.model_dump(mode="json"),
            "policy_artifact": "policy_selection.json" if spec.policy.selection else None,
            "research_only": True,
            "automatic_promotion": False,
            "limitations": list(spec.limitations),
            "evaluation_summary": material["summary"],
        }
        info["predictor_kind"] = spec.predictor.kind
        write_json(stage / "predictor.json", prepared.provenance["predictor"])
        write_json(stage / "release.json", info)
        write_json(stage / "contract.json", material["contract"])
        write_json(stage / "feature_recipe.json", material["recipe"])
        write_json(stage / "runtime.json", runtime_info())
        write_json(
            stage / "evidence.json",
            {
                "provenance": material["provenance"],
                "evaluation_spec": spec.model_dump(mode="json"),
                "selection_source_hashes": selection_source_hashes,
            },
        )
        descriptor = prepared.provenance["predictor"]
        names = ["config.json", *descriptor["files_sha256"]]
        for name in names:
            destination_file = inside(stage, name)
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared.run_dir / name, destination_file)
            expected = (
                prepared.provenance["parent_files_sha256"].get(name)
                if name == "config.json"
                else descriptor["files_sha256"][name]
            )
            if expected is None or sha256(destination_file) != expected:
                raise ValueError(f"copied predictor artifact mismatch: {name}")
        if prepared.calibration_dir is not None:
            shutil.copyfile(prepared.calibration_dir / "calibrator.json", stage / "calibrator.json")
            if (
                sha256(stage / "calibrator.json")
                != prepared.provenance["input_hashes"]["calibration/calibrator.json"]
            ):
                raise ValueError("copied calibrator mismatch")
        seal(stage)
        parity = _check_parity(ReleasedPredictor(stage), material, request.parity_atol)
        write_json(stage / "parity.json", parity)
        manifest = seal(stage)
        verify(stage)
        destination = root / manifest["release_id"]
        if destination.exists():
            raise FileExistsError("immutable release destination exists")
        stage.rename(destination)
        register_release(engine, manifest, info, destination)
        return {
            "release_id": manifest["release_id"],
            "artifact_dir": str(destination),
            "status": "succeeded",
            "parity": parity,
            "holdout_labels_loaded": False,
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    setup_logging()
    result = build_release(
        get_engine(),
        load_release_config(args.config),
        check_only=args.check,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
