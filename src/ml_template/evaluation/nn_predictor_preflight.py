"""Package 01: NN metadata check or explicit checkpoint verification; never evaluate."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PredictorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_run_id: str = Field(min_length=1)
    predictor_kind: Literal["final", "cv_mean"] = "final"
    database: str = "data/project.db"
    expected_silver_dataset_version_id: str | None = None
    expected_feature_set_id: str | None = None
    expected_split_id: str | None = None


def check_predictor(engine, request, *, verify_checkpoints=False, **roots):
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_nn_regression_predictor_inputs
    from ml_template.models.predictor_selection import load_selected_predictor

    spec = RegressionEvaluationConfig(
        parent_run_id=request.parent_run_id,
        predictor={"kind": request.predictor_kind},
        expected_silver_dataset_version_id=request.expected_silver_dataset_version_id,
        expected_feature_set_id=request.expected_feature_set_id,
        expected_split_id=request.expected_split_id,
    )
    prepared = prepare_nn_regression_predictor_inputs(engine, spec, **roots)
    result = {
        "ready": True,
        "package": "nn_lifecycle_01",
        "purpose": "predictor_only",
        "parent_run_id": request.parent_run_id,
        "predictor_kind": request.predictor_kind,
        "descriptor_schema": 3,
        "models_loaded": False,
        "predictions_computed": False,
        "labels_loaded": False,
        "holdout_labels_loaded": False,
        "files_written": False,
        "evaluation_enabled": False,
        "calibration_enabled": False,
        "release_enabled": False,
        "torch_version": prepared.provenance["versions"]["torch"],
        "predictor": prepared.provenance["predictor"],
    }
    if verify_checkpoints:
        model = load_selected_predictor(
            prepared.run_cfg, prepared.run_dir, prepared.provenance["predictor"]
        )
        result.update(model.verify_checkpoints())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("Duplicate predictor request key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    request = PredictorRequest.model_validate(
        yaml.load(args.config.read_text(), Loader=UniqueLoader)
    )
    from sqlalchemy import create_engine

    from ml_template.paths import ROOT

    path = Path(request.database).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Existing registry API gets a real SQLAlchemy engine, backed by SQLite mode=ro.
    engine = create_engine(
        "sqlite://", creator=lambda: sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    )
    try:
        print(
            json.dumps(
                check_predictor(engine, request, verify_checkpoints=args.verify_checkpoints),
                indent=2,
            )
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
