"""NN preparation/authorization and thin CLI; execution uses the shared evaluator."""

import argparse
import copy
import json
import sqlite3
from pathlib import Path

from ml_template.db import evaluation_registry as registry
from ml_template.tracking.nn_evaluation_contract import digest, read_document, validate_report


def rows(engine, table, sql, params=()):
    if table not in {
        "model_runs",
        "final_evaluations",
        "final_evaluation_metrics",
        "final_evaluation_locks",
    }:
        raise ValueError("Unknown registry table")
    with registry.connection(engine) as c:
        if not c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            return []
        cursor = c.execute(sql, params)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def is_nn_parent(engine, run_id):
    found = rows(
        engine, "model_runs", "SELECT model_type FROM model_runs WHERE run_id=?", (run_id,)
    )
    return len(found) == 1 and found[0]["model_type"] == "nn"


def cached(engine, identity):
    found = rows(
        engine,
        "final_evaluations",
        "SELECT * FROM final_evaluations WHERE active_identity=? AND status='succeeded'",
        (identity,),
    )
    if len(found) > 1:
        raise ValueError("Duplicate active evaluation identity")
    return found[0] if found else None


def record(engine, identity):
    found = rows(
        engine,
        "final_evaluations",
        "SELECT * FROM final_evaluations WHERE evaluation_id=?",
        (identity,),
    )
    return found[0] if len(found) == 1 else None


def exposure_warnings(engine, p):
    silver = p.spec.expected_silver_dataset_version_id
    result = [
        {"source": "evaluation_registry", **r}
        for r in rows(
            engine,
            "final_evaluations",
            "SELECT evaluation_id,holdout_fingerprint,accessed_at FROM final_evaluations WHERE silver_id=? AND accessed_at IS NOT NULL ORDER BY started_at,evaluation_id",
            (silver,),
        )
    ]
    result.extend(
        {
            "source": "legacy_or_parent_timestamp",
            "run_id": r["run_id"],
            "accessed_at": r["holdout_evaluated_at"],
        }
        for r in rows(
            engine,
            "model_runs",
            "SELECT run_id,holdout_evaluated_at FROM model_runs WHERE silver_dataset_version_id=? AND holdout_evaluated_at IS NOT NULL ORDER BY started_at DESC",
            (silver,),
        )
    )
    return result


def prepare_nn_evaluation(engine, spec, **roots):
    unit = spec.target_unit
    if (
        not isinstance(unit, str)
        or not unit.strip()
        or unit.strip() in {"target units", "target_units"}
    ):
        raise ValueError("NN evaluation requires explicit target units")
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_nn_regression_predictor_inputs
    from ml_template.evaluation.regression_postprocessing import attach

    raw = RegressionEvaluationConfig.model_validate(
        {
            **spec.model_dump(mode="json"),
            "calibration_id": None,
            "release_variant": "raw",
            "uncertainty": {"enabled": False},
        }
    )
    p = prepare_nn_regression_predictor_inputs(engine, raw, **roots)
    if p.provenance["versions"].get("torch") in {None, "not-installed"}:
        raise RuntimeError(
            "Install the pytorch extra before NN evaluation; no holdout access occurred"
        )
    resolved = {
        **spec.model_dump(mode="json"),
        **{
            k: getattr(p.spec, k)
            for k in [
                "expected_silver_dataset_version_id",
                "expected_feature_set_id",
                "expected_split_id",
            ]
        },
    }
    p.spec = RegressionEvaluationConfig.model_validate(resolved)
    p.provenance = copy.deepcopy(p.provenance)
    if p.spec.calibration_id is not None:
        attach(engine, p.spec, p.provenance)
    p.provenance["nn_evaluation_schema"] = 1
    p.identity_hash = digest({"spec": p.spec.model_dump(mode="json"), "provenance": p.provenance})
    p.purpose = "nn_evaluation"
    return p


def load_authorized_holdout(engine, p, evaluation_id):
    cfg = p.spec
    item = record(engine, evaluation_id)
    locks = rows(
        engine,
        "final_evaluation_locks",
        "SELECT evaluation_id FROM final_evaluation_locks WHERE silver_id=? AND evaluation_id=?",
        (cfg.expected_silver_dataset_version_id, evaluation_id),
    )
    if (
        getattr(p, "purpose", None) != "nn_evaluation"
        or not item
        or item["status"] != "running"
        or (not item.get("accessed_at"))
        or (item["parent_run_id"] != cfg.parent_run_id)
        or (item["identity_hash"] != p.identity_hash)
        or (item["active_identity"] != p.identity_hash)
        or (item["holdout_fingerprint"] != p.fingerprint)
        or (len(locks) != 1)
        or (json.loads(item["spec_json"]) != cfg.model_dump(mode="json"))
    ):
        raise ValueError("NN holdout requires committed matching access and active dataset lock")
    expected = (
        p.assignments.loc[p.assignments.partition.eq("holdout"), p.key_col]
        .sort_values(kind="stable")
        .tolist()
    )
    if p.hold_keys != expected:
        raise ValueError("NN holdout key inventory differs")
    from ml_template.features.prepared_recipe import load_partition

    X, y = load_partition(p, expected)
    return (X, y.to_numpy(float))


def verify_cache(engine, item, identity):
    data = read_document(item["artifact_dir"])
    stored = rows(
        engine,
        "final_evaluation_metrics",
        "SELECT result_json FROM final_evaluation_metrics WHERE evaluation_id=?",
        (item["evaluation_id"],),
    )
    validate_report(data, item, [json.loads(r["result_json"]) for r in stored])
    if item["identity_hash"] != identity:
        raise ValueError("Cached NN identity differs")
    return data


def validate_output(p, root):
    from ml_template.paths import ARTIFACTS_RUNS

    root = Path(root).resolve()
    if any(
        root.is_relative_to(Path(v).resolve())
        for v in [p.run_dir, ARTIFACTS_RUNS, p.parquet_path.parent, *p.roots.values()]
    ):
        raise ValueError("NN output cannot be inside source/run artifacts")
    if any(
        (parent / name).is_file()
        for parent in [root, *root.parents]
        for name in [
            "artifact_manifest.json",
            "manifest.json",
            "artifacts.json",
            "release.json",
        ]
    ):
        raise ValueError("NN output cannot be inside sealed artifacts")


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
        result = {}
        for k, v in node.value:
            key = loader.construct_object(k, deep=deep)
            if key in result:
                raise ValueError("Duplicate evaluation YAML key")
            result[key] = loader.construct_object(v, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    from sqlalchemy import create_engine

    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.paths import DATABASE

    spec = RegressionEvaluationConfig.model_validate(
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
        if not is_nn_parent(engine, spec.parent_run_id):
            raise ValueError("NN CLI requires registered NN parent")
        print(
            json.dumps(
                evaluate_spec(engine, spec, check_only=args.check, output_root=args.output_root),
                indent=2,
                allow_nan=False,
            )
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
