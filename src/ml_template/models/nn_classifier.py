"""Binary NN adapter using the existing supplied .ubj artifact paths.

Files contain tensor checkpoints plus embedded Gold state, not UBJSON. The
factory chooses the adapter by config, never by filename extension.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.models.nn.data import binary_labels, resolve_preprocessing
from ml_template.models.nn.metrics import probabilities
from ml_template.models.nn.params import NNParams

CHECKPOINT_VERSION = 1


def _engine():
    try:
        from ml_template.models.nn import engine
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch required: uv sync --extra nn; use uv run --extra nn"
            ) from exc
        raise
    return engine


class NNAdapter:
    def __init__(self, params: NNParams, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("NN seed must be a nonnegative integer")
        self.params, self.seed = params, int(seed)
        self.max_rounds = params.max_epochs
        self.model, self.preprocessor, self.best_epoch_ = None, None, None
        self.history_, self.execution_ = [], {}

    def _train(
        self, X, y, n_epochs, *, X_val=None, y_val=None, early_stopping=False, curve_metric=None
    ):
        self.model, self.preprocessor, self.best_epoch_ = None, None, None
        self.history_, self.execution_ = [], {}
        from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

        labels = binary_labels(X, y)
        val_labels = binary_labels(X_val, y_val) if X_val is not None else None
        pre = NNPreprocessor(resolve_preprocessing(self.params, X)).fit(X)
        inputs = pre.transform_arrays(X)
        validation = (pre.transform_arrays(X_val), val_labels) if X_val is not None else None
        result = _engine().train(
            inputs,
            labels,
            self.params,
            self.seed,
            n_epochs,
            validation=validation,
            early_stopping=early_stopping,
            curve_metric=curve_metric,
        )
        self.preprocessor, self.model = pre, result.model
        self.best_epoch_, self.history_, self.execution_ = (
            result.best_epoch,
            result.history,
            result.execution,
        )
        return result.best_epoch

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        return int(
            self._train(
                X_train, y_train, self.max_rounds, X_val=X_val, y_val=y_val, early_stopping=True
            )
        )

    def refit(self, X, y, n_rounds):
        if isinstance(n_rounds, bool) or int(n_rounds) != n_rounds or n_rounds < 1:
            raise ValueError("refit rounds must be a positive epoch count")
        self._train(X, y, int(n_rounds))

    def _require_fitted(self):
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("NNAdapter is not fitted")

    def predict_margin(self, X):
        self._require_fitted()
        return _engine().predict_logits(
            self.model, self.preprocessor.transform_arrays(X), self.params, self.seed
        )

    def predict_proba(self, X):
        return probabilities(self.predict_margin(X))

    def estimator_feature_names(self):
        self._require_fitted()
        return list(self.preprocessor.output_columns)

    def learning_curve(self, X_train, y_train, X_val, y_val, n_rounds, metric):
        if metric not in ("logloss", "auc", "aucpr", "error"):
            raise ValueError("unknown learning-curve metric")
        temp = NNAdapter(self.params, self.seed)
        temp._train(X_train, y_train, int(n_rounds), X_val=X_val, y_val=y_val, curve_metric=metric)
        table = pd.DataFrame(temp.history_)
        return table.loc[
            table.metric.eq(metric), ["round", "dataset", "metric", "value"]
        ].reset_index(drop=True)

    def save(self, path: Path):
        self._require_fitted()
        engine = _engine()
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "params": self.params.model_dump(mode="json"),
            "seed": self.seed,
            "preprocessor": self.preprocessor.state_dict(),
            "n_numeric": len(self.preprocessor.numeric_names),
            "cardinalities": list(self.preprocessor.cardinalities),
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "best_epoch": self.best_epoch_,
            "history": self.history_,
            "execution": self.execution_,
        }
        with Path(path).open("xb") as stream:
            engine.torch.save(payload, stream)

    @classmethod
    def load(cls, params, seed, path):
        from ml_template.models.nn.backbones import build_backbone
        from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

        engine = _engine()
        raw = engine.torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "checkpoint_version",
            "params",
            "seed",
            "preprocessor",
            "n_numeric",
            "cardinalities",
            "state_dict",
            "best_epoch",
            "history",
            "execution",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("invalid NN checkpoint fields")
        if (
            type(raw["checkpoint_version"]) is not int
            or raw["checkpoint_version"] != CHECKPOINT_VERSION
        ):
            raise ValueError("unsupported NN checkpoint version")
        saved_params = NNParams.model_validate(raw["params"])
        if (
            saved_params.model_dump() != params.model_dump()
            or type(raw["seed"]) is not int
            or raw["seed"] != seed
        ):
            raise ValueError("NN checkpoint configuration/seed differs from run")
        pre = NNPreprocessor.from_state_dict(raw["preprocessor"])
        if raw["n_numeric"] != len(pre.numeric_names) or raw["cardinalities"] != list(
            pre.cardinalities
        ):
            raise ValueError("NN checkpoint/preprocessor dimensions differ")
        if type(raw["best_epoch"]) is not int or raw["best_epoch"] < 1:
            raise ValueError("invalid NN checkpoint epoch")
        if not isinstance(raw["state_dict"], dict) or any(
            not isinstance(v, engine.torch.Tensor)
            or not bool(engine.torch.isfinite(v).all().item())
            for v in raw["state_dict"].values()
        ):
            raise ValueError("invalid/nonfinite NN weights")
        obj = cls(params, seed)
        with engine.execution_context(params, seed, "cpu"):
            model = build_backbone(len(pre.numeric_names), pre.cardinalities, params)
            model.load_state_dict(raw["state_dict"], strict=True)
        obj.model, obj.preprocessor = model.eval().cpu(), pre
        obj.best_epoch_, obj.history_, obj.execution_ = (
            raw["best_epoch"],
            raw["history"],
            raw["execution"],
        )
        return obj
