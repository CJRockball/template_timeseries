"""NN release evidence adapter/CLI; reuses shared sealed release construction."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from ml_template.evaluation.nn_evaluate import is_nn_parent


def neural_config(cfg):
    return (
        getattr(getattr(cfg, "model", None), "type", None) == "nn"
        and getattr(cfg, "task", None) == "regression"
    )


def evidence(engine, request):
    from ml_template.evaluation.nn_evaluate import prepare_nn_evaluation, record, verify_cache
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.features.prepared_recipe import verify_prediction_code
    from ml_template.scoring.release_artifacts import runtime_info

    item = record(engine, request.evaluation_id)
    if not item or item["status"] != "succeeded":
        raise ValueError("Explicit successful NN evaluation required")
    data = verify_cache(engine, item, item["identity_hash"])
    spec = RegressionEvaluationConfig.model_validate(data["spec"])
    if spec.parent_run_id != request.parent_run_id or request.calibration_id != spec.calibration_id:
        raise ValueError("NN release request differs from evaluated parent/calibration")
    p = prepare_nn_evaluation(engine, spec)
    old = data["provenance"]
    if (
        not neural_config(p.run_cfg)
        or p.spec != spec
        or p.provenance["predictor"] != old["predictor"]
        or (p.provenance["input_hashes"] != old["input_hashes"])
        or (p.provenance["feature_preparation"] != old["feature_preparation"])
        or (p.provenance["versions"] != old["versions"])
        or (p.provenance["python"] != old["python"])
        or (data["summary"]["n_rows"] != len(p.hold_keys))
    ):
        raise ValueError("Evaluated NN predictor/data/preprocessing/runtime changed")
    verify_prediction_code(old["code_hashes"], runtime_info()["code_sha256"])
    return (p, Path(item["artifact_dir"]), spec, data["summary"], old)


def validate_packaged_nn(cfg, info, contract, descriptor, evidence):
    neural = neural_config(cfg)
    declared = info.get("model_type") == "nn" or descriptor.get("model_type") == "nn"
    if not neural:
        if declared:
            raise ValueError("NN release metadata conflicts with model configuration")
        return
    spec = evidence["evaluation_spec"]
    summary = evidence["evaluation_summary"]
    if (
        cfg.fitted_features
        or info.get("model_type") != "nn"
        or descriptor.get("schema_version") != 3
        or (
            descriptor.get("serialization")
            != {
                "format": "torch",
                "weights_only": True,
                "checkpoint_version": 1,
                "preprocessing": "embedded",
            }
        )
        or (summary.get("model_type") != "nn")
        or (summary.get("nn_evaluation_schema") != 1)
        or (info.get("research_only") is not True)
        or (info.get("automatic_promotion") is not False)
        or (info.get("evaluation_summary") != summary)
    ):
        raise ValueError("NN package model/research/evaluation semantics differ")
    unit = spec.get("target_unit")
    if (
        not isinstance(unit, str)
        or not unit.strip()
        or unit.strip() in {"target_units", "target units"}
        or (info.get("target_unit") != unit)
        or (summary.get("target_unit") != unit)
    ):
        raise ValueError("NN release target units differ")
    if (
        contract.get("schema_version") != 3
        or contract.get("model_schema_stage") != "pre_gold"
        or contract.get("gold_preprocessing") != "owned_by_saved_model_adapter"
        or (contract.get("estimator_validation") != "saved_nn_adapter_finite_arrays")
        or (contract.get("unseen_category_policy") != "error")
    ):
        raise ValueError("NN release must retain strict pre-Gold input contract")


def validate_destination(p, root):
    from ml_template.paths import ARTIFACTS_RUNS

    root = Path(root).resolve()
    if any(
        root.is_relative_to(Path(source).resolve())
        for source in [p.run_dir, ARTIFACTS_RUNS, p.parquet_path.parent, *p.roots.values()]
    ):
        raise ValueError("NN release output cannot be inside source/run artifacts")
    if any(
        (parent / name).is_file()
        for parent in [root, *root.parents]
        for name in [
            "manifest.json",
            "artifact_manifest.json",
            "artifacts.json",
            "release.json",
        ]
    ):
        raise ValueError("NN release output cannot be inside sealed artifacts")


def compare_reference(reference, whole, chunked, single, key, columns, atol):
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (float, int))
        or (not np.isfinite(atol))
        or (not 0 < atol <= 1e-05)
    ):
        raise ValueError("Parity tolerance must be explicit, finite and at most 1e-5")
    for frame in [reference, whole, chunked]:
        if frame[key].isna().any() or frame[key].duplicated().any() or frame.empty:
            raise ValueError("Parity keys must be nonempty, unique and nonmissing")
        kinds = set()
        for value in frame[key]:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("Parity keys must be integer or string")
            kinds.add(type(value))
        if len(kinds) != 1:
            raise ValueError("Mixed parity key types")
    r = reference.set_index(key).sort_index()
    w = whole.set_index(key).sort_index()
    c = chunked.set_index(key).sort_index()
    if not r.index.equals(w.index) or not r.index.equals(c.index):
        raise ValueError("Parity key coverage differs")
    errors = {}
    for source, destination in columns.items():
        a = r[source].to_numpy()
        b = w[destination].to_numpy()
        z = c[destination].to_numpy()
        if any(v.dtype.kind not in "iuf" or not np.isfinite(v).all() for v in [a, b, z]):
            raise ValueError("Parity requires finite real continuous values")
        error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        errors[destination] = error
        chunk_error = float(np.max(np.abs(b.astype(float) - z.astype(float))))
        if error > atol or chunk_error > atol:
            raise ValueError(
                f"NN release parity failed: {destination}; reference_max_abs={error:.9g}; chunk_shuffle_max_abs={chunk_error:.9g}; atol={atol:.9g}"
            )
    if len(single) != 1 or single[key].iloc[0] != whole[key].iloc[0]:
        raise ValueError("Singleton parity key differs")
    for destination in columns.values():
        value = single[destination].iloc[0]
        target = whole[destination].iloc[0]
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or abs(float(value) - float(target)) > atol
        ):
            raise ValueError("NN singleton parity failed")
    return errors


def check_parity(predictor, material, atol):
    import pandas as pd

    from ml_template.evaluation.regression_postprocessing import reference_columns
    from ml_template.scoring.release_artifacts import sha256

    p = material["prepared"]
    path = material["evaluation_dir"] / "holdout_predictions.parquet"
    if sha256(path) != material["reference_sha256"]:
        raise ValueError("Evaluation predictions changed")
    extra = reference_columns(material)
    wanted = list(dict.fromkeys([p.key_col, "y_pred", "y_pred_raw", *extra]))
    reference = pd.read_parquet(path, columns=wanted)
    if (
        reference[p.key_col].isna().any()
        or reference[p.key_col].duplicated().any()
        or set(reference[p.key_col]) != set(p.hold_keys)
    ):
        raise ValueError("NN parity reference population differs")
    batch = material["source"].loc[reference[p.key_col]].reset_index()
    whole = predictor.predict(batch)
    chunked = predictor.predict(
        batch.iloc[::-1].reset_index(drop=True), chunk_size=max(1, min(137, len(batch) // 3))
    )
    single = predictor.predict(batch.iloc[:1])
    columns = {
        "y_pred_raw": "y_pred_raw",
        "y_pred": "y_pred_release",
        **{name: name for name in extra if name != "y_pred_raw"},
    }
    errors = compare_reference(reference, whole, chunked, single, p.key_col, columns, atol)
    if sha256(path) != material["reference_sha256"]:
        raise ValueError("Evaluation predictions changed during parity")
    return {
        "n_rows": len(reference),
        "atol": atol,
        "max_abs_diff": max(errors.values()),
        "column_max_abs_diff": errors,
        "keyed": True,
        "labels_loaded": False,
        "metrics_recomputed": False,
        "shuffled_chunk_parity": True,
        "singleton_parity": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        out = {}
        for a, b in node.value:
            key = loader.construct_object(a, deep=deep)
            if key in out:
                raise ValueError("Duplicate NN release YAML key")
            out[key] = loader.construct_object(b, deep=deep)
        return out

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    from sqlalchemy import create_engine

    from ml_template.paths import DATABASE
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig
    from ml_template.scoring.release_builder import build_release

    request = RegressionReleaseConfig.model_validate(
        yaml.load(args.config.read_text(), Loader=UniqueLoader)
    )
    path = (args.database if args.database is not None else DATABASE).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    db_mode = "ro" if args.check else "rw"
    engine = create_engine(
        "sqlite://", creator=lambda: sqlite3.connect(path.as_uri() + "?mode=" + db_mode, uri=True)
    )
    try:
        if not is_nn_parent(engine, request.parent_run_id):
            raise ValueError("NN release CLI requires registered NN parent")
        print(
            json.dumps(
                build_release(engine, request, check_only=args.check, output_root=args.output_root),
                indent=2,
                allow_nan=False,
            )
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
