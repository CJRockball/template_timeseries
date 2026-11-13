"""Prediction-only runtime: no data-registry access, labels or fitting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_template.evaluation.decision_policy import apply_policy, probabilities
from ml_template.evaluation.evaluation_config import PolicyConfig
from ml_template.features.feature_recipe import replay_recipe
from ml_template.models.predictor_selection import load_selected_predictor
from ml_template.scoring.input_contract import coerce_frame, validate_batch
from ml_template.scoring.pre_gold_schema import get_preprocessor_schema
from ml_template.scoring.release_artifacts import inside, read_json, verify, verify_runtime


class ReleasedPredictor:
    def __init__(self, directory, *, expected_release_id=None):
        self.directory = Path(directory).resolve()
        manifest = verify(self.directory, expected_release_id)
        verify_runtime(read_json(self.directory / "runtime.json"))
        self.release_id = manifest["release_id"]
        self.manifest = manifest
        self.info = read_json(self.directory / "release.json")
        if self.info.get("schema_version") != 1:
            raise ValueError("unsupported release schema")
        self.contract = read_json(self.directory / "contract.json")
        self.recipe = read_json(self.directory / "feature_recipe.json")
        self.preprocessor_input_schema = get_preprocessor_schema(self.contract)
        self.policy = PolicyConfig.model_validate(self.info["policy"])
        if [x["name"] for x in self.contract["source_schema"]] != self.recipe["raw_columns"]:
            raise ValueError("source schema differs from recipe")
        if [x["name"] for x in self.preprocessor_input_schema] != self.recipe["columns"]:
            raise ValueError("model schema differs from recipe")
        from ml_template.training.run_config import RunConfig

        cfg = RunConfig.model_validate(read_json(self.directory / "config.json"))
        descriptor = (
            read_json(self.directory / "predictor.json")
            if "predictor.json" in manifest["files"]
            else None
        )
        if descriptor is None and bool(cfg.fitted_features) != (
            "model.transformer.joblib" in manifest["files"]
        ):
            raise ValueError("fitted-transformer presence mismatch")
        if descriptor is not None:
            if descriptor["kind"] != self.info.get("predictor_kind", "final"):
                raise ValueError("release predictor identity differs from packaged descriptor")
            binding = read_json(self.directory / "evidence.json")
            evaluated_descriptor = binding.get("provenance", {}).get("predictor")
            if evaluated_descriptor is not None and evaluated_descriptor != descriptor:
                raise ValueError("release predictor differs from evaluated member identity")
            if descriptor["kind"] == "cv_mean" and evaluated_descriptor is None:
                raise ValueError("fold release lacks fold-average evaluation evidence")
            evaluated_kind = (
                binding.get("evaluation_spec", {}).get("predictor", {}).get("kind", "final")
            )
            if evaluated_kind != descriptor["kind"]:
                raise ValueError("release predictor kind differs from evaluated specification")
            if descriptor["kind"] == "cv_mean" and (
                self.info["calibration_id"] is not None
                or self.info["release_variant"] != "raw"
                or self.policy.kind != "none"
            ):
                raise ValueError("cv_mean only supports uncalibrated probabilities with no policy")
        self.calibrator = None
        if self.info["calibration_id"]:
            from ml_template.models.probability_calibrator import ProbabilityCalibrator

            if "calibrator.json" not in manifest["files"]:
                raise ValueError("missing release calibrator")
            self.calibrator = ProbabilityCalibrator.load(self.directory / "calibrator.json")
        elif "calibrator.json" in manifest["files"]:
            raise ValueError("unexpected calibrator")
        if self.info["release_variant"] not in {"raw", "calibrated"}:
            raise ValueError("unsupported release score variant")
        if self.info["release_variant"] == "calibrated" and self.calibrator is None:
            raise ValueError("calibrated release has no calibrator")
        if self.policy.selection is not None:
            from ml_template.evaluation.policy_selection import validate_selection_record

            binding = read_json(self.directory / "evidence.json")
            record = read_json(inside(self.directory, "policy_selection.json"))
            spec = SimpleNamespace(
                policy=self.policy,
                parent_run_id=self.info["parent_run_id"],
                calibration_id=self.info["calibration_id"],
                release_variant=self.info["release_variant"],
            )
            validate_selection_record(
                spec, record, binding["provenance"], binding["selection_source_hashes"]
            )
            if (
                manifest["files"].get("calibrator.json", {}).get("sha256")
                != record["source_hashes"]["calibrator.json"]
            ):
                raise ValueError("policy was selected with a different calibrator")
            if (
                manifest["files"]["policy_selection.json"]["sha256"]
                != self.policy.selection.artifact_sha256
            ):
                raise ValueError("selected policy artifact hash mismatch")
        # Integrity/runtime verification precedes joblib deserialisation in load_adapter.
        self.model = load_selected_predictor(cfg, self.directory, descriptor)

    def _prepare_source_features(self, source):
        """Internal path for already-validated source rows; fitted features stay in adapter."""
        return coerce_frame(replay_recipe(source, self.recipe), self.preprocessor_input_schema)

    def prepare_features(self, frame):
        """Validate canonical keyed inputs and replay frozen stateless features.

        Returns selected pre-transform inputs indexed by entity key. Do not apply the
        saved fitted transformer here: the model adapter owns that boundary.
        """
        return self._prepare_source_features(validate_batch(frame, self.contract))

    def predict(self, frame, *, chunk_size=None, include_policy=True):
        raw = validate_batch(frame, self.contract)
        if chunk_size is None:
            chunk_size = len(raw)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        pieces = []
        for start in range(0, len(raw), chunk_size):
            source = raw.iloc[start : start + chunk_size]
            features = self._prepare_source_features(source)
            scores = probabilities(self.model.predict_proba(features))
            if len(scores) != len(source):
                raise ValueError("model prediction length mismatch")
            pieces.append(scores)
        p_raw = probabilities(np.concatenate(pieces))
        data = {self.contract["key_col"]: raw.index.to_numpy(), "p_raw": p_raw}
        if self.calibrator is not None:
            data["p_calibrated"] = probabilities(self.calibrator.transform(p_raw))
            if len(data["p_calibrated"]) != len(p_raw):
                raise ValueError("calibrator prediction length mismatch")
        selected = data[f"p_{self.info['release_variant']}"]
        data["p_release"] = selected
        out = pd.DataFrame(data)
        if include_policy and self.policy.kind != "none":
            order = np.argsort(raw.index.to_numpy(), kind="stable")
            flags = np.empty(len(out), dtype=bool)
            flags[order] = apply_policy(selected[order], self.policy)
            out["policy_flag"] = flags
        out["release_id"] = self.release_id
        return out

    def __new__(cls, directory=None, *, expected_release_id=None):
        if cls is ReleasedPredictor and directory is not None:
            verify(directory, expected_release_id)
            info = read_json(Path(directory) / "release.json")
            if info.get("task") == "regression":
                from ml_template.scoring.regression_predictor import RegressionReleasedPredictor

                return RegressionReleasedPredictor(
                    directory, expected_release_id=expected_release_id
                )
        return object.__new__(cls)
