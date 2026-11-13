"""Standalone -I worker; imports only the selected PINNED ml_template at execution.

This script is copied outside the pinned source tree. It requires no new methods
in old adapters and never edits models, manifests, source, registries or labels.
"""

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def native_contributions(adapter, features):
    import numpy as np
    import xgboost as xgb

    transformer = getattr(adapter, "transformer", None)
    expanded = transformer.transform(features) if transformer is not None else features.copy()
    inner = getattr(adapter, "inner", adapter)
    booster = inner._booster()
    if json.loads(booster.save_config())["learner"]["objective"]["name"] != "reg:squarederror":
        raise ValueError("only identity-link reg:squarederror is supported")
    if list(expanded.columns) != booster.feature_names or not expanded.index.equals(features.index):
        raise ValueError("expanded estimator schema or entity alignment differs")
    best = booster.attr("best_iteration")
    rounds = (0, int(best) + 1) if best is not None else (0, 0)
    matrix = xgb.DMatrix(expanded, enable_categorical=True)
    native = np.asarray(
        booster.predict(
            matrix,
            pred_contribs=True,
            approx_contribs=False,
            strict_shape=True,
            iteration_range=rounds,
        )
    )
    if (
        native.shape != (len(expanded), 1, len(expanded.columns) + 1)
        or not np.isfinite(native).all()
    ):
        raise ValueError("unexpected native continuous contribution shape/values")
    return expanded, native[:, 0, :-1].astype(float), native[:, 0, -1].astype(float)


def run(request, output, *, check_only):
    import numpy as np
    import pandas as pd

    import ml_template
    from ml_template.scoring.release_artifacts import (
        file_entries,
        read_json,
        sha256,
        verify,
        verify_runtime,
    )

    source_root = Path(ml_template.__file__).resolve().parent
    if source_root != Path(request["runtime_source_root"]).resolve():
        raise ValueError("selected Python imports a different pinned source root")
    if Path(__file__).resolve().is_relative_to(source_root):
        raise ValueError("standalone worker must remain outside pinned source")
    binding = request["binding"]
    release, score = Path(request["release_dir"]), Path(request["scoring_dir"])
    runtime = read_json(release / "runtime.json")

    def reverify():
        manifest = verify(release, binding["release_id"])
        verify_runtime(runtime)
        if manifest["content_sha256"] != binding["release_content_sha256"]:
            raise ValueError("release changed")
        if (score / "artifacts.json").is_symlink():
            raise ValueError("scoring manifest symlink unsupported")
        if (
            sha256(score / "artifacts.json") != binding["scoring_manifest_sha256"]
            or file_entries(score, exclude=("artifacts.json",)) != binding["scoring_files"]
        ):
            raise ValueError("scoring artifacts changed")

    reverify()
    info, summary = read_json(release / "release.json"), read_json(score / "summary.json")
    if (
        info.get("task") != "regression"
        or info.get("release_variant") not in {"raw", "recalibrated"}
        or (info.get("calibration_id") is not None and not info.get("postprocessing"))
        or info["policy"]["kind"] != "none"
        or summary.get("task") != "regression"
        or summary.get("schema_version") != 2
        or summary["release_id"] != binding["release_id"]
        or summary["scoring_batch_id"] != binding["scoring_batch_id"]
    ):
        raise ValueError("unsupported or mismatched continuous batch/release")
    contract = read_json(release / "contract.json")
    key = contract["key_col"]
    from ml_template.scoring.input_contract import validate_batch

    snapshot = score / binding["snapshot_name"]
    frame = (
        pd.read_csv(snapshot, dtype={key: "string"} if contract["key_kind"] == "str" else None)
        if snapshot.suffix == ".csv"
        else pd.read_parquet(snapshot)
    )
    raw = validate_batch(frame, contract)
    saved = pd.read_parquet(score / "predictions.parquet")
    interval_names = [
        name
        for item in info.get("interval_columns", {}).values()
        for name in (item["lower"], item["upper"])
    ]
    required = {
        key,
        "y_pred_raw",
        "y_pred_release",
        "release_id",
        "scoring_batch_id",
        *interval_names,
    }
    if (
        saved.columns.duplicated().any()
        or set(saved) != required
        or saved[key].isna().any()
        or saved[key].duplicated().any()
        or set(saved[key]) != set(raw.index)
        or len(saved) != len(raw)
        or len(raw) != summary["n_rows"]
        or not 0 < len(raw) <= request["max_rows"]
    ):
        raise ValueError("invalid saved regression prediction coverage/schema")
    saved = saved.set_index(key).loc[raw.index]
    for name in ("release_id", "scoring_batch_id"):
        if not saved[name].eq(binding[name]).all():
            raise ValueError("saved prediction identity differs")
    for name in ("y_pred_raw", "y_pred_release", *interval_names):
        values = saved[name]
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or not np.isfinite(values.to_numpy(float)).all()
        ):
            raise ValueError("saved continuous predictions must be finite numeric")
    response = {
        "schema_version": 2,
        "ready": True,
        "task": "regression",
        "release_id": binding["release_id"],
        "scoring_batch_id": binding["scoring_batch_id"],
        "binding_sha256": binding["binding_sha256"],
        "runtime_source_root": str(source_root),
        "runtime_sha256": fingerprint(runtime),
        "worker_sha256": sha256(Path(__file__)),
        "labels_loaded": False,
        "model_deserialized": False,
        "predictions_computed": False,
        "n_rows": len(raw),
    }
    if check_only:
        reverify()
        return response
    from ml_template.models.model_factory import load_adapter
    from ml_template.models.predictor_selection import safe_file, verify_predictor_files
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.training.run_config import RunConfig

    cfg = RunConfig.model_validate(read_json(release / "config.json"))
    descriptor = read_json(release / "predictor.json")
    predictor = ReleasedPredictor(release, expected_release_id=binding["release_id"])
    reproduced = (
        predictor.predict(frame, chunk_size=request["chunk_size"]).set_index(key).loc[raw.index]
    )
    tolerance = request["tolerance"]
    atol, rtol = tolerance["prediction_atol"], tolerance["prediction_rtol"]
    for name in ("y_pred_raw", "y_pred_release", *interval_names):
        if not np.allclose(saved[name], reproduced[name], atol=atol, rtol=rtol):
            raise ValueError("saved continuous prediction parity failed: " + name)
    feature_parts, phi_parts, base_parts, varying = [], [], [], set()
    names = None
    for start in range(0, len(raw), request["chunk_size"]):
        selected = predictor.prepare_features(frame.iloc[start : start + request["chunk_size"]])
        first, total, baseline = None, None, None
        for member in descriptor["members"]:
            verify_predictor_files(release, descriptor)
            adapter = load_adapter(cfg, safe_file(release, member["model_path"]))
            expanded, phi, base = native_contributions(adapter, selected)
            if names is None:
                names = list(expanded.columns)
            if list(expanded.columns) != names:
                raise ValueError("member estimator feature names differ")
            if first is None:
                first = expanded.copy()
                total, baseline = np.zeros_like(phi), np.zeros_like(base)
            else:
                for name in names:
                    if not first[name].equals(expanded[name]):
                        varying.add(name)
            total += phi * member["weight"]
            baseline += base * member["weight"]
            del adapter
        feature_parts.append(first)
        phi_parts.append(total)
        base_parts.append(baseline)
    features = pd.concat(feature_parts)
    for name in varying:
        features[name] = "varies across fold transforms"
    phi, bases = np.concatenate(phi_parts), np.concatenate(base_parts)
    postprocessing = info.get("postprocessing")
    if postprocessing is not None:
        correction = postprocessing["parameters"]["correction"]
        phi = phi * correction["slope"]
        bases = bases * correction["slope"] + correction["intercept"]
    predicted = reproduced[
        "y_pred_release" if postprocessing is not None else "y_pred_raw"
    ].to_numpy(float)
    if not np.allclose(bases + phi.sum(axis=1), predicted, atol=atol, rtol=rtol):
        raise ValueError("weighted contributions do not reconstruct released predictor")
    reverify()
    features.to_parquet(output / "features.parquet", index=True)
    np.savez_compressed(
        output / "contributions.npz", contributions=phi, base_values=bases, predictions=predicted
    )
    reproduced.to_parquet(output / "reproduced_predictions.parquet", index=True)
    import xgboost as xgb

    response.update(
        model_deserialized=True,
        predictions_computed=True,
        method="xgboost_native_tree_shap",
        output_scale="prediction_target_units",
        feature_dependence="tree_path_dependent",
        n_features=len(names),
        feature_names=names,
        varying_feature_values=sorted(varying),
        predictor_kind=descriptor["kind"],
        postprocessing=info.get("postprocessing"),
        interval_columns=info.get("interval_columns", {}),
        explained_prediction_column="y_pred_release",
        aggregation="saved_member_weights_sum_signed_contributions_and_baselines",
        xgboost_version=xgb.__version__,
        files=file_entries(output, exclude=("request.json", "response.json")),
    )
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        response = run(json.loads(args.request.read_text()), args.output, check_only=args.check)
    except Exception as exc:
        (args.output / "response.json").write_text(json.dumps({"ready": False, "error": str(exc)}))
        raise
    (args.output / "response.json").write_text(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
