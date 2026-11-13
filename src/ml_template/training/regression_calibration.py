"""Audited postprocessing fitted on reserved rows, not dev/OOF/final holdout."""

import argparse
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from ml_template.models.regression_postprocessing import (
    correct,
    coverage_tag,
    fit_correction,
    fit_quantiles,
    levels,
    rank_for,
    validate_parameters,
)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    parent_run_id: str = Field(min_length=1)
    model_type: Literal["xgb", "nn"] = "xgb"
    predictor_kind: str = "final"
    correction: str = "none"
    coverages: tuple[float, ...] = (0.8, 0.9, 0.95)
    target_unit: str = Field(min_length=1)
    min_calibration_rows: int = Field(default=30, ge=1)
    min_recalibration_rows: int = Field(default=10, ge=2)
    intended_use: str = Field(min_length=1)
    acknowledge_prior_exposure: bool = False
    prior_exposure_note: str = ""

    @model_serializer(mode="wrap")
    def _legacy_serialization(self, handler):
        result = handler(self)
        if self.model_type == "xgb":
            result.pop("model_type", None)
        return result

    @model_validator(mode="after")
    def check(self):
        if self.predictor_kind not in {"final", "cv_mean"} or self.correction not in {
            "none",
            "bias",
            "affine",
        }:
            raise ValueError("unsupported predictor/correction")
        levels(self.coverages)
        if not self.coverages and self.correction == "none":
            raise ValueError("request intervals or a correction")
        if not self.target_unit.strip() or not self.intended_use.strip():
            raise ValueError("document units and intended use")
        if self.acknowledge_prior_exposure and not self.prior_exposure_note.strip():
            raise ValueError("prior exposure acknowledgement requires a note")
        return self


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prediction_binding(provenance):
    critical = {
        name: value
        for name, value in provenance["code_hashes"].items()
        if name.startswith(("models/", "features/", "preprocessing/", "training/"))
        or name in {"evaluation/regression_inputs.py", "evaluation/regression_postprocessing.py"}
    }
    return {
        "predictor": provenance["predictor"],
        "input_hashes": {
            k: v for k, v in provenance["input_hashes"].items() if k != "regression_calibration"
        },
        "feature_preparation": provenance["feature_preparation"],
        "code_hashes": critical,
        "versions": provenance["versions"],
        "python": provenance["python"],
    }


def prior_exposures(engine, silver_id):
    from ml_template.db.evaluation_registry import connection

    with connection(engine) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='final_evaluations'"
        ).fetchone():
            return []
        cursor = conn.execute(
            "SELECT evaluation_id,holdout_fingerprint,accessed_at FROM final_evaluations WHERE silver_id=? AND accessed_at IS NOT NULL ORDER BY started_at,evaluation_id",
            (silver_id,),
        )
        return [
            dict(zip([item[0] for item in cursor.description], row, strict=True))
            for row in cursor.fetchall()
        ]


def ensure_tables(engine):
    from ml_template.db.evaluation_registry import connection
    from ml_template.db.evaluation_registry import ensure_tables as ensure_evaluations

    ensure_evaluations(engine)
    with connection(engine) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS regression_calibrations (
          calibration_id TEXT PRIMARY KEY,parent_run_id TEXT NOT NULL,status TEXT NOT NULL,
          artifact_dir TEXT NOT NULL,manifest_sha256 TEXT,started_at TEXT NOT NULL,
          accessed_at TEXT,finished_at TEXT,request_json TEXT NOT NULL,prior_json TEXT NOT NULL,
          error_summary TEXT)""")
        conn.commit()


def get_record(engine, identity):
    from ml_template.db.evaluation_registry import connection

    with connection(engine) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='regression_calibrations'"
        ).fetchone():
            raise ValueError("no registered regression calibration")
        cursor = conn.execute(
            "SELECT * FROM regression_calibrations WHERE calibration_id=?", (identity,)
        )
        row = cursor.fetchone()
        return (
            dict(zip([item[0] for item in cursor.description], row, strict=True)) if row else None
        )


def load_calibration(engine, identity, parent_run_id, provenance, target_unit):
    from ml_template.monitoring.artifacts import digest, read_json, verify

    record = get_record(engine, identity)
    if not record or record["status"] != "succeeded" or record["parent_run_id"] != parent_run_id:
        raise ValueError("explicit successful calibration of this parent required")
    root = Path(record["artifact_dir"])
    verify(root, record["manifest_sha256"])
    state = read_json(root / "calibration.json")
    payload = {k: v for k, v in state.items() if k != "payload_sha256"}
    if (
        fingerprint(payload) != state["payload_sha256"]
        or state["calibration_id"] != identity
        or state["config"] != json.loads(record["request_json"])
        or state["binding"] != prediction_binding(provenance)
        or state["config"]["target_unit"] != target_unit
        or state["config"]["parent_run_id"] != parent_run_id
    ):
        raise ValueError("calibration predictor/data/code/config/unit binding differs")
    parameters = validate_parameters(state["parameters"])
    if set(parameters["quantiles"]) != {coverage_tag(v) for v in state["config"]["coverages"]}:
        raise ValueError("calibrated levels differ from frozen request")
    if parameters["correction"]["method"] != state["config"]["correction"]:
        raise ValueError("correction differs from frozen request")
    return {
        "calibration_id": identity,
        "artifact_dir": str(root),
        "state": state,
        "sha256": digest(root / "calibration.json"),
    }


def _prepare_calibration_inputs(engine, config, spec, roots):
    if getattr(config, "model_type", "xgb") == "nn":
        from ml_template.evaluation.regression_inputs import prepare_nn_regression_predictor_inputs

        prepared = prepare_nn_regression_predictor_inputs(engine, spec, **roots)
        if prepared.purpose != "predictor_only":
            raise ValueError("NN calibration requires verified predictor-only preparation")
        return prepared
    from ml_template.evaluation.regression_inputs import prepare_regression

    return prepare_regression(engine, spec, **roots)


def _role_inventory(prepared):
    a, key = prepared.assignments, prepared.key_col
    if (
        a.columns.duplicated().any()
        or not {key, "partition"} <= set(a)
        or a[key].isna().any()
        or a[key].duplicated().any()
        or not a.partition.isin(["dev", "holdout", "calibration", "recalibration"]).all()
    ):
        raise ValueError("Invalid or overlapping reserved-role key inventory")
    return {
        role: a.loc[a.partition.eq(role), key].tolist()
        for role in ("dev", "holdout", "calibration", "recalibration")
    }


def calibration_roles(prepared, config):
    inventory = _role_inventory(prepared)
    roles = {}
    for role, required, minimum in (
        ("calibration", bool(config.coverages), config.min_calibration_rows),
        ("recalibration", config.correction != "none", config.min_recalibration_rows),
    ):
        keys = inventory[role]
        if required and len(keys) < minimum:
            raise ValueError(
                "reserve sufficient "
                + role
                + " rows BEFORE training; old dev/holdout rows are ineligible"
            )
        roles[role] = keys if required else []
    for coverage in config.coverages:
        if rank_for(len(roles["calibration"]), coverage) > len(roles["calibration"]):
            raise ValueError("sample too small for a finite requested interval")
    return roles


def _load_reserved_partition(prepared, role, keys):
    if role not in {"calibration", "recalibration"}:
        raise ValueError("Only reserved calibration roles may be loaded")
    keys = list(keys)
    inventory = _role_inventory(prepared)
    if not keys or len(keys) != len(set(keys)) or set(keys) != set(inventory[role]):
        raise ValueError("Requested keys must match exactly the declared reserved role")
    if set(keys) & (set(inventory["dev"]) | set(inventory["holdout"])):
        raise ValueError("Development/holdout keys are forbidden for postprocessing fitting")
    from ml_template.features.prepared_recipe import load_partition

    return load_partition(prepared, keys)


def calibrate(
    engine,
    config,
    *,
    output_root=None,
    check_only=True,
    silver_root=None,
    split_root=None,
    feature_root=None,
):
    from ml_template.db.evaluation_registry import connection
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.features.prepared_recipe import verify_prepared
    from ml_template.models.predictor_selection import load_evaluation_predictor
    from ml_template.monitoring.artifacts import seal, write_json

    spec = RegressionEvaluationConfig(
        parent_run_id=config.parent_run_id, predictor={"kind": config.predictor_kind}
    )
    roots = {
        name: value
        for name, value in {
            "silver_root": silver_root,
            "split_root": split_root,
            "feature_root": feature_root,
        }.items()
        if value is not None
    }
    prepared = _prepare_calibration_inputs(engine, config, spec, roots)
    silver_id = prepared.run_cfg.data.silver_dataset_version_id
    prior = prior_exposures(engine, silver_id)
    roles = calibration_roles(prepared, config)
    report = {
        "ready": not prior or config.acknowledge_prior_exposure,
        "parent_run_id": config.parent_run_id,
        "n_calibration": len(roles["calibration"]),
        "n_recalibration": len(roles["recalibration"]),
        "prior_exposures": prior,
        "holdout_labels_loaded": False,
        "calibration_labels_loaded": False,
        "files_written": False,
    }
    if getattr(config, "model_type", "xgb") == "nn":
        report.update(
            model_type="nn",
            scope="postprocessing_only",
            models_loaded=False,
            predictions_computed=False,
            recalibration_labels_loaded=False,
            evaluation_enabled=False,
            release_enabled=False,
        )
    if check_only:
        return report
    if not report["ready"]:
        raise ValueError(
            "holdout already exposed; explicit research acknowledgement and note required"
        )
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "regression_calibrations"
    output_root = Path(output_root).resolve()
    for source in (
        prepared.run_dir,
        prepared.parquet_path.parent,
        prepared.roots["split_root"],
        prepared.roots["feature_root"],
    ):
        source = Path(source).resolve()
        if output_root == source or output_root.is_relative_to(source):
            raise ValueError("calibration output cannot be inside a source artifact tree")
    for parent in (output_root, *output_root.parents):
        if any(
            (parent / name).is_file()
            for name in (
                "artifact_manifest.json",
                "manifest.json",
                "artifacts.json",
                "release.json",
            )
        ):
            raise ValueError("calibration output cannot be nested under sealed artifacts")
    identity = "regcal_" + uuid.uuid4().hex
    directory = output_root / identity
    ensure_tables(engine)
    with connection(engine) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO final_evaluation_locks (silver_id,evaluation_id) VALUES (?,?)",
                (silver_id, identity),
            )
            conn.execute(
                "INSERT INTO regression_calibrations (calibration_id,parent_run_id,status,artifact_dir,started_at,request_json,prior_json) VALUES (?,?,'running',?,?,?,?)",
                (
                    identity,
                    config.parent_run_id,
                    str(directory),
                    datetime.now(UTC).isoformat(),
                    config.model_dump_json(),
                    json.dumps(prior),
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    try:
        if prior_exposures(engine, silver_id) != prior:
            raise ValueError("holdout exposure history changed before calibration")
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "request.json", config.model_dump(mode="json"))
        with connection(engine) as conn:
            conn.execute(
                "UPDATE regression_calibrations SET accessed_at=? WHERE calibration_id=? AND status='running'",
                (datetime.now(UTC).isoformat(), identity),
            )
            conn.commit()
        model = load_evaluation_predictor(prepared)
        correction = {"method": "none", "slope": 1.0, "intercept": 0.0}
        if roles["recalibration"]:
            X, y = _load_reserved_partition(prepared, "recalibration", roles["recalibration"])
            correction = fit_correction(model.predict(X), y.to_numpy(float), config.correction)
        quantiles = {}
        if roles["calibration"]:
            X, y = _load_reserved_partition(prepared, "calibration", roles["calibration"])
            quantiles = fit_quantiles(
                correct(model.predict(X), correction), y.to_numpy(float), config.coverages
            )
        parameters = {
            "schema_version": 1,
            "correction": correction,
            "n_calibration": len(roles["calibration"]),
            "quantiles": quantiles,
        }
        validate_parameters(parameters)
        verify_prepared(prepared)
        fresh = _prepare_calibration_inputs(engine, config, spec, roots)
        if prediction_binding(fresh.provenance) != prediction_binding(prepared.provenance):
            raise ValueError("calibration inputs changed during fitting")
        state = {
            "schema_version": 1,
            "calibration_id": identity,
            "config": config.model_dump(mode="json"),
            "binding": prediction_binding(prepared.provenance),
            "partition_key_hashes": {role: fingerprint(keys) for role, keys in roles.items()},
            "parameters": parameters,
            "holdout_labels_loaded": False,
        }
        state["payload_sha256"] = fingerprint(state)
        write_json(directory / "calibration.json", state)
        manifest_hash = seal(directory)
        with connection(engine) as conn:
            cursor = conn.execute(
                "UPDATE regression_calibrations SET status='succeeded',finished_at=?,manifest_sha256=? WHERE calibration_id=? AND status='running'",
                (datetime.now(UTC).isoformat(), manifest_hash, identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("only a running calibration can complete")
            conn.execute("DELETE FROM final_evaluation_locks WHERE evaluation_id=?", (identity,))
            conn.commit()
    except Exception as exc:
        with connection(engine) as conn:
            conn.execute(
                "UPDATE regression_calibrations SET status='failed',finished_at=?,error_summary=? WHERE calibration_id=? AND status='running'",
                (datetime.now(UTC).isoformat(), str(exc)[:2000], identity),
            )
            conn.execute("DELETE FROM final_evaluation_locks WHERE evaluation_id=?", (identity,))
            conn.commit()
        raise
    return {
        **report,
        "calibration_id": identity,
        "artifact_dir": str(directory),
        "status": "succeeded",
        "calibration_labels_loaded": bool(roles["calibration"] or roles["recalibration"]),
        "files_written": True,
        **(
            {
                "models_loaded": True,
                "predictions_computed": True,
                "calibration_labels_loaded": bool(roles["calibration"]),
                "recalibration_labels_loaded": bool(roles["recalibration"]),
            }
            if getattr(config, "model_type", "xgb") == "nn"
            else {}
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--database", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    # Reject duplicate YAML keys before validating the request.
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate calibration YAML key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config = CalibrationConfig.model_validate(
        yaml.load(args.config.read_text(), Loader=UniqueLoader)
    )
    owned = config.model_type == "nn" or args.database is not None
    if owned:
        import sqlite3

        from sqlalchemy import create_engine

        from ml_template.paths import DATABASE

        database = (args.database if args.database is not None else DATABASE).expanduser().resolve()
        if not database.is_file():
            raise FileNotFoundError(database)
        mode = "ro" if args.check else "rw"
        engine = create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(database.as_uri() + "?mode=" + mode, uri=True),
        )
    else:
        from ml_template.db.connection import get_engine

        engine = get_engine()
    try:
        print(
            json.dumps(
                calibrate(engine, config, output_root=args.output_root, check_only=args.check),
                indent=2,
                allow_nan=False,
            )
        )
    finally:
        if owned:
            engine.dispose()


if __name__ == "__main__":
    main()
