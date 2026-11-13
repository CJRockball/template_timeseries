"""NN regression training/checkpoint adapter. Outputs are never probabilities."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.models.nn.data import resolve_preprocessing
from ml_template.models.nn.params import NNRegressionParams
from ml_template.models.nn.regression import targets

CHECKPOINT_VERSION = 1


def _engine():
    try:
        from ml_template.models.nn import regression_engine
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch required: uv sync --extra nn; use uv run --extra nn"
            ) from exc
        raise
    return regression_engine


class NNRegressionAdapter:
    def __init__(self, params: NNRegressionParams, seed: int):
        if not isinstance(params, NNRegressionParams):
            raise ValueError("NN regression requires NNRegressionParams")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("NN seed must be a nonnegative integer")
        self.params, self.seed = params, int(seed)
        self.max_rounds = params.max_epochs
        self.model, self.preprocessor, self.best_epoch_ = None, None, None
        self.history_, self.execution_ = [], {}
        self.early_stopping_history = None

    def _train(
        self, X, y, n_epochs, *, X_val=None, y_val=None, early_stopping=False, curve_metric=None
    ):
        self.model, self.preprocessor, self.best_epoch_ = None, None, None
        self.history_, self.execution_ = [], {}
        self.early_stopping_history = None
        labels = targets(X, y)
        validation_labels = targets(X_val, y_val) if X_val is not None else None
        from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

        pre = NNPreprocessor(resolve_preprocessing(self.params, X)).fit(X)
        inputs = pre.transform_arrays(X)
        validation = (pre.transform_arrays(X_val), validation_labels) if X_val is not None else None
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
        self.model, self.preprocessor, self.best_epoch_ = result.model, pre, result.best_epoch
        self.history_, self.execution_ = result.history, result.execution
        if early_stopping:
            table = pd.DataFrame(self.history_)
            table = table.loc[table.metric.eq(self.params.monitor)].copy()
            table["dataset"] = table.dataset.map({"train": "inner_fit", "val": "inner_stop"})
            self.early_stopping_history = table.assign(
                selected_round=self.best_epoch_, direction="min", curve_role="inner_early_stopping"
            )
        return result.best_epoch

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        return int(
            self._train(
                X_train, y_train, self.max_rounds, X_val=X_val, y_val=y_val, early_stopping=True
            )
        )

    def refit(self, X, y, n_rounds):
        if (
            isinstance(n_rounds, bool)
            or not isinstance(n_rounds, (int, np.integer))
            or n_rounds < 1
        ):
            raise ValueError("refit epochs must be a positive integer")
        self._train(X, y, int(n_rounds))

    def _require_fitted(self):
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("NNRegressionAdapter is not fitted")

    def predict(self, X):
        self._require_fitted()
        from ml_template.models.nn.engine import predict_logits

        result = np.asarray(
            predict_logits(
                self.model, self.preprocessor.transform_arrays(X), self.params, self.seed
            ),
            dtype=float,
        )
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError("invalid NN continuous predictions")
        return result

    def estimator_feature_names(self):
        self._require_fitted()
        return list(self.preprocessor.output_columns)

    def learning_curve(self, X_train, y_train, X_val, y_val, n_rounds, metric):
        if metric not in {"rmse", "mae"}:
            raise ValueError("regression curves require rmse or mae")
        if (
            isinstance(n_rounds, bool)
            or not isinstance(n_rounds, (int, np.integer))
            or n_rounds < 1
        ):
            raise ValueError("curve epochs must be a positive integer")
        adapter = type(self)(self.params, self.seed)
        adapter._train(
            X_train, y_train, int(n_rounds), X_val=X_val, y_val=y_val, curve_metric=metric
        )
        table = pd.DataFrame(adapter.history_)
        return table.loc[
            table.metric.eq(metric), ["round", "dataset", "metric", "value"]
        ].reset_index(drop=True)

    def save(self, path: Path):
        self._require_fitted()
        import torch

        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if any(
            v.dtype != torch.float32 or not bool(torch.isfinite(v).all().item())
            for v in state.values()
        ):
            raise ValueError("nonfinite NN regression weights")
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "task": "regression",
            "params": self.params.model_dump(mode="json"),
            "seed": self.seed,
            "preprocessor": self.preprocessor.state_dict(),
            "n_numeric": len(self.preprocessor.numeric_names),
            "cardinalities": list(self.preprocessor.cardinalities),
            "state_dict": state,
            "best_epoch": self.best_epoch_,
            "history": self.history_,
            "execution": self.execution_,
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, params, seed, path):
        import torch

        from ml_template.models.nn.backbones import build_backbone
        from ml_template.models.nn.engine import execution_context
        from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "checkpoint_version",
            "task",
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
            raise ValueError("invalid NN regression checkpoint fields")
        if (
            type(raw["checkpoint_version"]) is not int
            or raw["checkpoint_version"] != CHECKPOINT_VERSION
            or raw["task"] != "regression"
        ):
            raise ValueError("unsupported NN regression checkpoint version/task")
        saved = NNRegressionParams.model_validate(raw["params"])
        if (
            saved.model_dump() != params.model_dump()
            or type(raw["seed"]) is not int
            or raw["seed"] != seed
        ):
            raise ValueError("NN regression checkpoint configuration/seed differs")
        pre = NNPreprocessor.from_state_dict(raw["preprocessor"])
        if (
            type(raw["n_numeric"]) is not int
            or raw["n_numeric"] != len(pre.numeric_names)
            or raw["cardinalities"] != list(pre.cardinalities)
            or any(type(n) is not int for n in raw["cardinalities"])
        ):
            raise ValueError("NN regression checkpoint/preprocessor dimensions differ")
        if type(raw["best_epoch"]) is not int or raw["best_epoch"] < 1:
            raise ValueError("invalid checkpoint epoch")
        if not isinstance(raw["state_dict"], dict) or any(
            not isinstance(v, torch.Tensor)
            or v.dtype != torch.float32
            or not bool(torch.isfinite(v).all().item())
            for v in raw["state_dict"].values()
        ):
            raise ValueError("invalid/nonfinite NN regression weights")
        if not isinstance(raw["history"], list) or not isinstance(raw["execution"], dict):
            raise ValueError("invalid NN regression training metadata")
        obj = cls(params, seed)
        with execution_context(params, seed, "cpu"):
            model = build_backbone(len(pre.numeric_names), pre.cardinalities, params)
            model.load_state_dict(raw["state_dict"], strict=True)
        obj.model, obj.preprocessor = model.eval().cpu(), pre
        obj.best_epoch_, obj.history_, obj.execution_ = (
            raw["best_epoch"],
            raw["history"],
            raw["execution"],
        )
        return obj
