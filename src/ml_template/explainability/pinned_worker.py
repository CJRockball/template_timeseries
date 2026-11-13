"""Standalone inference worker launched by an explicit Python with -I.

Do NOT import the current explainability package here. The worker imports ml_template
only from the explicitly selected installed environment and verifies its exact saved runtime.
It never edits releases, model artifacts, source hashes, databases or evaluation history.
"""

import argparse
import hashlib
import json
from pathlib import Path


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_difference(expected, actual):
    code_old, code_new = expected.get("code_sha256", {}), actual.get("code_sha256", {})
    changed = sorted(
        name for name in set(code_old) | set(code_new) if code_old.get(name) != code_new.get(name)
    )
    packages_old, packages_new = expected.get("packages", {}), actual.get("packages", {})
    packages = {
        name: {"expected": packages_old.get(name), "actual": packages_new.get(name)}
        for name in set(packages_old) | set(packages_new)
        if packages_old.get(name) != packages_new.get(name)
    }
    return {
        "python_expected": expected.get("python"),
        "python_actual": actual.get("python"),
        "package_differences": packages,
        "changed_source_count": len(changed),
        "changed_source_first_20": changed[:20],
    }


def _aligned_saved(frame, saved, key):
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError("snapshot key coverage invalid")
    if saved[key].isna().any() or saved[key].duplicated().any():
        raise ValueError("saved prediction keys invalid")
    if len(frame) != len(saved) or set(frame[key]) != set(saved[key]):
        raise ValueError("saved predictions do not exactly cover snapshot")
    return saved.set_index(key).loc[frame[key].to_numpy()]


def run(request, output, *, check_only):
    import numpy as np
    import pandas as pd

    import ml_template

    source_root = Path(ml_template.__file__).resolve().parent
    if source_root != Path(request["runtime_source_root"]).resolve():
        raise ValueError("selected Python imports a different ml_template source root")
    worker_path = Path(__file__).resolve()
    if worker_path.is_relative_to(source_root):
        raise ValueError("worker must remain outside the pinned ml_template source tree")
    from ml_template.scoring.release_artifacts import (
        file_entries,
        read_json,
        runtime_info,
        sha256,
        verify,
        verify_runtime,
    )

    release_dir, score_dir = Path(request["release_dir"]), Path(request["scoring_dir"])
    binding = request["binding"]
    manifest = verify(release_dir, binding["release_id"])
    runtime = read_json(release_dir / "runtime.json")
    try:
        verify_runtime(runtime)
    except ValueError as exc:
        actual = runtime_info()
        details = _runtime_difference(runtime, actual)
        raise ValueError(f"pinned runtime mismatch: {details}") from exc
    if manifest["content_sha256"] != binding["release_content_sha256"]:
        raise ValueError("release changed after parent preflight")

    def verify_score_files():
        if (score_dir / "artifacts.json").is_symlink():
            raise ValueError("scoring manifest symlink unsupported")
        if sha256(score_dir / "artifacts.json") != binding["scoring_manifest_sha256"]:
            raise ValueError("saved scoring manifest changed")
        got = file_entries(score_dir, exclude=("artifacts.json",))
        if got != binding["scoring_files"]:
            raise ValueError("saved scoring files changed")

    verify_score_files()
    summary = read_json(score_dir / "summary.json")
    if (
        summary["scoring_batch_id"] != binding["scoring_batch_id"]
        or summary["release_id"] != binding["release_id"]
    ):
        raise ValueError("scoring identity changed")
    contract = read_json(release_dir / "contract.json")
    info = read_json(release_dir / "release.json")
    key = contract["key_col"]
    snapshot = score_dir / binding["snapshot_name"]
    if snapshot.suffix == ".csv":
        kwargs = {"dtype": {key: "string"}} if contract["key_kind"] == "str" else {}
        frame = pd.read_csv(snapshot, **kwargs)
    else:
        frame = pd.read_parquet(snapshot)
    from ml_template.scoring.input_contract import validate_batch

    raw = validate_batch(frame, contract)
    if len(raw) != summary["n_rows"] or len(raw) > request["max_rows"]:
        raise ValueError("row limit/count mismatch")
    expected_columns = {key, "p_raw", "p_release", "release_id", "scoring_batch_id"}
    if info["calibration_id"]:
        expected_columns.add("p_calibrated")
    if info["policy"]["kind"] != "none":
        expected_columns.add("policy_flag")
    saved_frame = pd.read_parquet(score_dir / "predictions.parquet")
    if saved_frame.columns.duplicated().any() or set(saved_frame.columns) != expected_columns:
        raise ValueError("saved prediction schema differs")
    saved = _aligned_saved(frame, saved_frame, key)
    for column, expected in (
        ("release_id", binding["release_id"]),
        ("scoring_batch_id", binding["scoring_batch_id"]),
    ):
        if not (saved[column] == expected).all():
            raise ValueError("saved prediction identity differs")
    if info["policy"]["kind"] == "alert_rate" and (
        request["batch_definition"] != info["policy"]["batch_definition"]
    ):
        raise ValueError("original decision batch acknowledgement required")
    response = {
        "schema_version": 1,
        "ready": True,
        "release_id": binding["release_id"],
        "scoring_batch_id": binding["scoring_batch_id"],
        "binding_sha256": binding["binding_sha256"],
        "runtime_source_root": str(source_root),
        "runtime_sha256": _fingerprint(runtime),
        "worker_sha256": sha256(worker_path),
        "n_rows": len(frame),
        "labels_loaded": False,
        "model_deserialized": False,
        "predictions_computed": False,
    }
    if check_only:
        verify_score_files()
        verify(release_dir, binding["release_id"])
        verify_runtime(runtime)
        return response

    from ml_template.features.feature_recipe import replay_recipe
    from ml_template.features.fitted_features import TransformingAdapter
    from ml_template.models.xgb_classifier import XGBAdapter
    from ml_template.scoring.input_contract import coerce_frame
    from ml_template.scoring.predictor import ReleasedPredictor

    predictor = ReleasedPredictor(release_dir, expected_release_id=binding["release_id"])
    # Full decision batch: never recompute top-k independently per explanation chunk.
    reproduced = predictor.predict(frame, chunk_size=request["chunk_size"]).set_index(key)
    if not reproduced.index.equals(raw.index):
        raise ValueError("reproduced prediction keys/order differ")
    tolerance = request["tolerance"]
    differences = {}
    for column in ("p_raw", "p_calibrated", "p_release"):
        if column not in saved:
            continue
        old = saved[column].to_numpy(dtype=float)
        new = reproduced[column].to_numpy(dtype=float)
        if not np.isfinite(old).all() or ((old < 0) | (old > 1)).any():
            raise ValueError("saved probabilities invalid")
        if not np.allclose(
            old, new, atol=tolerance["probability_atol"], rtol=tolerance["probability_rtol"]
        ):
            raise ValueError(f"saved-batch probability parity failed: {column}")
        differences[column] = float(np.max(np.abs(old - new)))
    if "policy_flag" in saved:
        if (
            not pd.api.types.is_bool_dtype(saved.policy_flag.dtype)
            or saved.policy_flag.isna().any()
        ):
            raise ValueError("saved policy flags invalid")
        if not np.array_equal(saved.policy_flag.to_numpy(), reproduced.policy_flag.to_numpy()):
            raise ValueError("saved full-batch policy parity failed")
    adapter = predictor.model
    if type(adapter) is TransformingAdapter:
        if adapter.transformer is None:
            raise ValueError("missing saved fitted transformer")
        estimator = adapter.inner
    else:
        estimator = adapter
    if type(estimator) is not XGBAdapter or estimator.model is not None:
        raise ValueError(
            "worker supports a saved native XGBAdapter, optionally TransformingAdapter"
        )
    booster = estimator._booster()
    if json.loads(booster.save_config())["learner"]["objective"]["name"] != "binary:logistic":
        raise ValueError("only binary:logistic explanations supported")
    import xgboost as xgb

    features, contribution_parts, margin_parts = [], [], []
    for start in range(0, len(raw), request["chunk_size"]):
        source = raw.iloc[start : start + request["chunk_size"]]
        selected = coerce_frame(replay_recipe(source, predictor.recipe), contract["model_schema"])
        expanded = (
            adapter.transformer.transform(selected)
            if type(adapter) is TransformingAdapter
            else selected
        )
        if list(expanded.columns) != booster.feature_names:
            raise ValueError("expanded estimator feature order differs")
        if not expanded.index.equals(source.index):
            raise ValueError("feature pipeline changed entity keys/order")
        matrix = xgb.DMatrix(expanded, enable_categorical=True)
        native = np.asarray(
            booster.predict(matrix, pred_contribs=True, approx_contribs=False, strict_shape=True)
        )
        margin = np.asarray(booster.predict(matrix, output_margin=True, strict_shape=True))
        if native.shape != (len(source), 1, expanded.shape[1] + 1) or margin.shape != (
            len(source),
            1,
        ):
            raise ValueError("unexpected native binary explanation shape")
        features.append(expanded)
        contribution_parts.append(native[:, 0, :])
        margin_parts.append(margin[:, 0])
    expanded = pd.concat(features)
    native = np.concatenate(contribution_parts)
    margins = np.concatenate(margin_parts)
    if not np.isfinite(native).all() or not np.isfinite(margins).all():
        raise ValueError("nonfinite explanation output")
    verify_score_files()
    verify(release_dir, binding["release_id"])
    verify_runtime(runtime)
    expanded.to_parquet(output / "features.parquet", index=True)
    np.savez_compressed(
        output / "contributions.npz",
        contributions=native[:, :-1],
        base_values=native[:, -1],
        margins=margins,
        p_raw=reproduced.p_raw.to_numpy(dtype=float),
    )
    reproduced.to_parquet(output / "reproduced_predictions.parquet", index=True)
    fitted_metadata = {}
    if type(adapter) is TransformingAdapter:
        for spec, step in zip(adapter.specs, adapter.transformer.steps, strict=True):
            for name in step.output_columns():
                fitted_metadata[name] = {"kind": "fitted_feature", "step": spec}
    response.update(
        {
            "model_deserialized": True,
            "predictions_computed": True,
            "n_features": expanded.shape[1],
            "feature_names": list(expanded.columns),
            "fitted_feature_metadata": fitted_metadata,
            "method": "xgboost_native_tree_shap",
            "feature_dependence": "tree_path_dependent",
            "output_scale": "base_model_log_odds",
            "saved_probability_max_abs_diff": differences,
            "policy_parity": "policy_flag" in saved,
            "files": file_entries(output, exclude=("request.json", "response.json")),
            "xgboost_version": xgb.__version__,
        }
    )
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    try:
        response = run(request, args.output, check_only=args.check)
    except Exception as exc:
        (args.output / "response.json").write_text(
            json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        )
        raise
    (args.output / "response.json").write_text(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
