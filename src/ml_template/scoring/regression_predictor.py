"""Registry-free, sealed regression inference with canonical keyed input."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.evaluation.evaluation_config import PolicyConfig
from ml_template.models.regression_predictor import continuous
from ml_template.scoring.input_contract import validate_batch
from ml_template.scoring.pre_gold_schema import get_preprocessor_schema
from ml_template.scoring.release_artifacts import read_json, sha256, verify, verify_runtime


class RegressionReleasedPredictor:
    def __init__(self, directory, *, expected_release_id=None):
        self.directory = Path(directory).resolve()
        self.manifest = verify(self.directory, expected_release_id)
        verify_runtime(read_json(self.directory / "runtime.json"))
        self.release_id = self.manifest["release_id"]
        self.info = read_json(self.directory / "release.json")
        if (
            self.info.get("schema_version") != 2
            or self.info.get("task") != "regression"
            or self.info.get("prediction_kind") != "continuous"
        ):
            raise ValueError("Unsupported regression release semantics")
        if self.info["release_variant"] not in {"raw", "recalibrated"}:
            raise ValueError("unsupported regression point variant")
        self.policy = PolicyConfig.model_validate(self.info["policy"])
        if self.policy.kind != "none":
            raise ValueError("Classification policy is invalid for regression")
        self.calibrator = None
        self.contract = read_json(self.directory / "contract.json")
        self.recipe = read_json(self.directory / "feature_recipe.json")
        self.preprocessor_input_schema = get_preprocessor_schema(self.contract)
        from ml_template.models.predictor_selection import load_selected_predictor
        from ml_template.training.run_config import RunConfig

        cfg = RunConfig.model_validate(read_json(self.directory / "config.json"))
        if cfg.task != "regression":
            raise ValueError("Expected regression release config")
        from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

        validate_regression_specs(cfg.fitted_features)
        descriptor = read_json(self.directory / "predictor.json")
        evidence = read_json(self.directory / "evidence.json")
        provenance, spec = (evidence["provenance"], evidence["evaluation_spec"])
        if (
            descriptor != provenance["predictor"]
            or descriptor["kind"] != self.info["predictor_kind"]
        ):
            raise ValueError("Packaged/evaluated predictor differs")
        if (
            spec.get("task") != "regression"
            or spec["parent_run_id"] != self.info["parent_run_id"]
            or spec["predictor"]["kind"] != descriptor["kind"]
        ):
            raise ValueError("Release evaluation binding differs")
        summary = evidence["evaluation_summary"]
        if (
            summary["evaluation_id"] != self.info["evaluation_id"]
            or summary["parent_run_id"] != self.info["parent_run_id"]
            or summary.get("calibration_id") != self.info["calibration_id"]
            or summary.get("release_variant") != self.info["release_variant"]
        ):
            raise ValueError("Release summary identity differs")
        from ml_template.evaluation.evaluation_inputs import digest

        expected_identity = digest({"spec": spec, "provenance": provenance})
        if expected_identity != summary["identity_hash"]:
            raise ValueError("Evaluation evidence digest differs")
        if sha256(self.directory / "config.json") != provenance["input_hashes"]["config"]:
            raise ValueError("Packaged run config differs from evaluated config")
        outputs = {"raw": "y_pred_raw", "release": "y_pred_release"}
        if self.info.get("prediction_columns") != outputs:
            raise ValueError("Unexpected regression output schema")
        from ml_template.features.prepared_recipe import validate_packaged

        feature_info = None
        if "features.json" in self.manifest["files"]:
            if sha256(self.directory / "features.json") != provenance["input_hashes"]["features"]:
                raise ValueError("Packaged feature metadata differs from evaluated input schema")
            feature_info = read_json(self.directory / "features.json")
        self.recipe = validate_packaged(
            self.recipe,
            self.contract,
            self.preprocessor_input_schema,
            cfg,
            provenance,
            feature_info,
        )
        from ml_template.evaluation.regression_postprocessing import load_packaged

        self.postprocessing = load_packaged(self.directory, self.info, provenance, spec)
        from ml_template.scoring.nn_release import validate_packaged_nn

        validate_packaged_nn(cfg, self.info, self.contract, descriptor, evidence)
        self.model = load_selected_predictor(cfg, self.directory, descriptor)

    def _prepare_source_features(self, source):
        from ml_template.features.prepared_recipe import prepare_frame

        return prepare_frame(source, self.recipe, self.preprocessor_input_schema)

    def prepare_features(self, frame):
        return self._prepare_source_features(validate_batch(frame, self.contract))

    def predict(self, frame, *, chunk_size=None, include_policy=True):
        verify(self.directory, self.release_id)
        verify_runtime(read_json(self.directory / "runtime.json"))
        raw = validate_batch(frame, self.contract)
        chunk_size = len(raw) if chunk_size is None else chunk_size
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        pieces = []
        for start in range(0, len(raw), chunk_size):
            features = self._prepare_source_features(raw.iloc[start : start + chunk_size])
            pieces.append(continuous(self.model.predict(features), len(features)))
        values = continuous(np.concatenate(pieces), len(raw))
        result = pd.DataFrame(
            {
                self.contract["key_col"]: raw.index.to_numpy(),
                "y_pred_raw": values,
                "y_pred_release": values.copy(),
                "release_id": self.release_id,
            }
        )
        from ml_template.evaluation.regression_postprocessing import add_predictions

        return add_predictions(result, self.postprocessing)
