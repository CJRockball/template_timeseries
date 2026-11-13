"""Typed run configuration: schema, loading, overrides and sweep expansion.

One ``RunConfig`` describes one model run: parameters and data references only.
Launch identity (``experiment_run_id``) and ``run_id`` are assigned by the runner,
so they never change the config hash. The ``analysis`` section is also left out of
the hash, because it changes what is reported, not the model.

Metrics: the standard set (see ``evaluation.metrics``) is always computed. The
``names`` and ``alert_rates`` fields only add extras; they cannot remove anything.

Relative paths inside configs (``model_file``, sweep ``base``, ``dataset_config``)
resolve against the current directory first, then the repository root.

Check a config without running anything:
    python -m ml_template.training.run_config --config configs/experiment/xgb_baseline.yaml
    python -m ml_template.training.run_config --sweep configs/sweep/xgb_grid.yaml
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from ml_template.evaluation.metrics import resolved_keys, with_standard
from ml_template.models.nn.params import NNModelConfig, NNRegressionParams
from ml_template.paths import ROOT

SCHEMA_VERSION = 2
MAX_SWEEP_RUNS = 200
HASH_EXCLUDED_SECTIONS = {"analysis"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    dataset_name: str  # folder under data/silver/, e.g. "ai4i"
    dataset_config: str  # dataset contract YAML, e.g. configs/ai4i_binary.yaml
    silver_dataset_version_id: str
    feature_set_id: str | None = None  # None = Silver feature-role columns only


class SplitConfig(_Strict):
    """Holdout policy and development-only CV-fold policy."""

    method: Literal["stratified", "random"] = "stratified"
    holdout_fraction: float = Field(0.2, gt=0, lt=1)
    n_folds: int = Field(5, ge=2)

    # Controls development/holdout membership. Keep fixed.
    seed: int = 42

    # Controls development-fold assignments only.
    # None preserves the original behavior and split identity.
    cv_seed: int | None = None
    calibration_fraction: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    recalibration_fraction: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _reserved_roles(self):
        if (self.calibration_fraction or self.recalibration_fraction) and (
            self.method != "random"
            or self.holdout_fraction + self.calibration_fraction + self.recalibration_fraction >= 1
        ):
            raise ValueError(
                "reserved calibration roles require random splitting and positive development share"
            )
        return self

    @model_serializer(mode="wrap")
    def _legacy_serialization(self, handler):
        data = handler(self)
        for name in ("calibration_fraction", "recalibration_fraction"):
            if data.get(name) == 0:
                data.pop(name, None)
        return data


class TrainingConfig(_Strict):
    """Trainer settings that are not model parameters."""

    # Share of each fold's training rows held back to decide when to stop.
    validation_fraction: float = Field(0.2, gt=0, lt=0.5)
    # Final rounds = median best round across folds x this factor.
    refit_rounds_factor: float = Field(1.0, ge=0.5, le=2.0)


class MetricsConfig(_Strict):
    primary: str = "pr_auc"  # a result key, e.g. "pr_auc" or "recall_at_5pct"
    names: list[str] = Field(default_factory=list)  # extras; standard metrics always run
    alert_rates: list[float] = Field(default_factory=list)  # extras; 5% and 10% always run

    @model_validator(mode="before")
    @classmethod
    def _add_standard(cls, data: Any) -> Any:
        if isinstance(data, dict):
            names, rates = with_standard(data.get("names"), data.get("alert_rates"))
            data = {**data, "names": names, "alert_rates": rates}
        return data

    @model_validator(mode="after")
    def _check(self) -> MetricsConfig:
        keys = resolved_keys(self.names, self.alert_rates)  # raises on unknown/bad
        if self.primary not in keys:
            raise ValueError(f"primary {self.primary!r} is not one of {keys}")
        return self


class AnalysisConfig(_Strict):
    """Standard per-run analyses. Excluded from the config hash."""

    permutation_importance: bool = True
    permutation_repeats: int = Field(10, ge=1)
    permutation_max_rows: int = Field(50_000, ge=100)  # validation rows used per fold
    error_sample_size: int = Field(25, ge=0)  # rows per kind in error_sample.csv; 0 = off
    learning_curves: bool = False  # per-round train/validation metric table; off by default
    learning_curve_metric: Literal["logloss", "aucpr", "auc", "error", "rmse", "mae"] = "logloss"
    learning_curve_rounds: int | None = Field(None, ge=1)  # None = twice each fold's best round


class XGBParams(_Strict):
    objective: Literal["binary:logistic"] = "binary:logistic"
    eval_metric: Literal["aucpr", "logloss", "auc"] = "aucpr"  # watched by early stopping
    n_estimators: int = Field(2000, ge=1)  # a cap; early stopping picks the real count
    early_stopping_rounds: int = Field(50, ge=1)
    max_depth: int = Field(4, ge=1)
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    min_child_weight: float = Field(1.0, ge=0)
    reg_lambda: float = Field(1.0, ge=0)
    reg_alpha: float = Field(0.0, ge=0)
    max_bin: int = Field(256, ge=2)
    # "auto" = negatives / positives of the rows being fitted.
    scale_pos_weight: Annotated[float, Field(gt=0)] | Literal["auto"] = "auto"
    device: Literal["cpu", "cuda"] = "cpu"


class XGBRegressionParams(_Strict):
    objective: Literal["reg:squarederror"] = "reg:squarederror"
    eval_metric: Literal["rmse", "mae"] = "rmse"
    n_estimators: int = Field(2000, ge=1)
    early_stopping_rounds: int = Field(50, ge=1)
    max_depth: int = Field(4, ge=1)
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    min_child_weight: float = Field(1.0, ge=0)
    reg_lambda: float = Field(1.0, ge=0)
    reg_alpha: float = Field(0.0, ge=0)
    max_bin: int = Field(256, ge=2)
    device: Literal["cpu", "cuda"] = "cpu"


class RegressionMetricsConfig(_Strict):
    primary: Literal["rmse", "mae", "r2"] = "rmse"
    names: list[str] = Field(default_factory=lambda: ["rmse", "mae", "r2"])
    alert_rates: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_regression(self):
        if set(self.names) - {"rmse", "mae", "r2"} or self.alert_rates:
            raise ValueError("Regression supports rmse/mae/r2 and no alert rates")
        if self.primary not in self.names:
            raise ValueError("Primary regression metric must be in names")
        return self


class XGBModelConfig(_Strict):
    type: Literal["xgb"] = "xgb"
    params: XGBParams | XGBRegressionParams = Field(default_factory=XGBParams)


class LGBMParams(_Strict):
    objective: Literal["binary"] = "binary"
    eval_metric: Literal["average_precision", "binary_logloss", "auc"] = "average_precision"
    n_estimators: int = Field(2000, ge=1)  # a cap; early stopping picks the real count
    early_stopping_rounds: int = Field(50, ge=1)
    num_leaves: int = Field(15, ge=2)
    max_depth: int = Field(-1, ge=-1)  # -1 = no limit
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)
    subsample_freq: int = Field(1, ge=0)  # bagging happens only when this is above 0
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    min_child_samples: int = Field(20, ge=1)
    reg_lambda: float = Field(1.0, ge=0)
    reg_alpha: float = Field(0.0, ge=0)
    max_bin: int = Field(255, ge=2)
    # "auto" = negatives / positives of the rows being fitted.
    scale_pos_weight: Annotated[float, Field(gt=0)] | Literal["auto"] = "auto"


class LGBMModelConfig(_Strict):
    type: Literal["lgbm"] = "lgbm"
    params: LGBMParams = Field(default_factory=LGBMParams)


class CatBoostParams(_Strict):
    loss_function: Literal["Logloss"] = "Logloss"
    eval_metric: Literal["PRAUC", "Logloss", "AUC"] = "PRAUC"  # watched by early stopping
    n_estimators: int = Field(2000, ge=1)  # a cap; early stopping picks the real count
    early_stopping_rounds: int = Field(50, ge=1)
    depth: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)  # Bernoulli row sampling
    rsm: float = Field(0.8, gt=0, le=1)  # share of features per split
    l2_leaf_reg: float = Field(3.0, ge=0)
    border_count: int = Field(254, ge=1, le=65535)
    scale_pos_weight: Annotated[float, Field(gt=0)] | Literal["auto"] = "auto"


class CatBoostModelConfig(_Strict):
    type: Literal["catboost"] = "catboost"
    params: CatBoostParams = Field(default_factory=CatBoostParams)


ModelConfig = Annotated[
    XGBModelConfig | LGBMModelConfig | CatBoostModelConfig | NNModelConfig,
    Field(discriminator="type"),
]


class RunConfig(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    task: Literal["binary_classification", "regression"] = "binary_classification"
    experiment_name: str
    seed: int = 42  # model seed; the split has its own
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    metrics: MetricsConfig | RegressionMetricsConfig = Field(default_factory=MetricsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    # Steps that learn from training rows (group statistics, power transforms); None = none.
    fitted_features: list[dict[str, Any]] | None = None
    # Add further model types to ``ModelConfig`` (a union keyed on ``type``).
    model: ModelConfig = Field(default_factory=XGBModelConfig)

    @model_validator(mode="before")
    @classmethod
    def _regression_defaults(cls, data):
        if isinstance(data, dict) and data.get("task") == "regression":
            data = copy.deepcopy(data)
            data["metrics"] = RegressionMetricsConfig.model_validate(data.get("metrics", {}))
            raw_model = data.get("model", {})
            if isinstance(raw_model, dict) and raw_model.get("type") == "nn":
                params = raw_model.setdefault("params", {})
                if isinstance(params, dict):
                    params.setdefault("task", "regression")
            raw_params = (
                raw_model.get("params", {}) if isinstance(raw_model, dict) else raw_model.params
            )
            monitor = (
                raw_params.get("eval_metric", raw_params.get("monitor", "rmse"))
                if isinstance(raw_params, dict)
                else getattr(raw_params, "eval_metric", getattr(raw_params, "monitor", "rmse"))
            )
            analysis = {
                "permutation_importance": False,
                "error_sample_size": 0,
                "learning_curves": False,
                "learning_curve_metric": monitor,
            }
            given = data.get("analysis", {})
            if not isinstance(given, dict):
                given = given.model_dump()
            analysis.update(given)
            data["analysis"] = analysis
            if "split" not in data:
                data["split"] = {"method": "random"}
        return data

    @model_validator(mode="after")
    def _task_compatibility(self):
        regression = self.task == "regression"
        if regression:
            xgb = self.model.type == "xgb" and isinstance(self.model.params, XGBRegressionParams)
            neural = self.model.type == "nn" and isinstance(self.model.params, NNRegressionParams)
            if not (xgb or neural):
                raise ValueError(
                    "Regression requires XGB regression parameters or NN regression parameters"
                )
            if neural and self.fitted_features:
                raise ValueError("NN regression owns preprocessing; leave fitted_features unset")
            if self.split.method != "random":
                raise ValueError("Regression requires split.method: random")
            if not isinstance(self.metrics, RegressionMetricsConfig):
                raise ValueError("Regression requires regression metrics")
            if self.metrics.primary == "r2":
                raise ValueError("Use rmse or mae as the initial primary metric; r2 is diagnostic")
            from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

            validate_regression_specs(self.fitted_features)
            if self.analysis.permutation_importance or self.analysis.error_sample_size:
                raise ValueError("Regression analyses are not implemented; disable them")
        else:
            if isinstance(self.metrics, RegressionMetricsConfig):
                raise ValueError("Binary tasks require classification metrics")
            if self.model.type == "xgb" and isinstance(self.model.params, XGBRegressionParams):
                raise ValueError("Regression objective requires task: regression")
            if self.model.type == "nn" and isinstance(self.model.params, NNRegressionParams):
                raise ValueError("NN regression parameters require task: regression")
            if self.split.method != "stratified":
                raise ValueError("Binary task requires stratified splits")
        return self

    @field_validator("experiment_name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", v):
            raise ValueError("experiment_name may only use letters, digits, _ . -")
        return v

    @field_validator("fitted_features")
    @classmethod
    def _fitted(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v:
            from ml_template.features.fitted_features import validate_specs

            validate_specs(v)
        return v

    @model_validator(mode="after")
    def _curve_metric_compatibility(self):
        if self.analysis.learning_curves:
            allowed = (
                {"rmse", "mae"}
                if self.task == "regression"
                else {"logloss", "aucpr", "auc", "error"}
            )
            if self.analysis.learning_curve_metric not in allowed:
                raise ValueError(f"Learning-curve metric incompatible with task {self.task!r}")
        return self


# --- raw YAML handling -------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    """Absolute path, or relative to cwd if it exists there, else to the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = ROOT / p
    return candidate if candidate.exists() else p


def _read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: top level must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted}: {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """'model.params.max_depth=6' -> ('model.params.max_depth', 6)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value: {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _load_raw(path: str | Path) -> dict[str, Any]:
    """Read an experiment YAML. ``model_file`` is merged under ``model``."""
    raw = _read_yaml(path)
    model_file = raw.pop("model_file", None)
    if model_file:
        raw["model"] = _deep_merge(_read_yaml(model_file), raw.get("model", {}))
    return raw


# --- public API --------------------------------------------------------------


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load one experiment YAML, apply ``key=value`` overrides, validate."""
    raw = _load_raw(path)
    for item in overrides or []:
        key, value = parse_override(item)
        _set_dotted(raw, key, value)
    return RunConfig.model_validate(raw)


def expand_sweep(path: str | Path) -> list[RunConfig]:
    """Expand a sweep YAML into one validated RunConfig per grid point.

    Sweep file keys: ``base`` (experiment YAML), optional ``experiment_name``,
    and ``grid`` mapping dotted keys to non-empty lists of values.
    """
    sweep = _read_yaml(path)
    unknown = set(sweep) - {"base", "experiment_name", "grid"}
    if unknown:
        raise ValueError(f"unknown sweep keys: {sorted(unknown)}")
    if "base" not in sweep or "grid" not in sweep:
        raise ValueError("sweep file needs 'base' and 'grid'")
    grid: dict[str, list[Any]] = sweep["grid"]
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("every grid entry must be a non-empty list")
    total = 1
    for values in grid.values():
        total *= len(values)
    if total > MAX_SWEEP_RUNS:
        raise ValueError(f"grid has {total} runs; limit is {MAX_SWEEP_RUNS}")

    base = _load_raw(sweep["base"])
    if "experiment_name" in sweep:
        base["experiment_name"] = sweep["experiment_name"]
    keys = list(grid)
    configs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        raw = copy.deepcopy(base)
        for key, value in zip(keys, combo, strict=True):
            _set_dotted(raw, key, value)
        configs.append(RunConfig.model_validate(raw))
    return configs


def config_json(cfg: RunConfig) -> str:
    """Canonical JSON snapshot, stored with each run for exact replay."""
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=2)


def config_hash(cfg: RunConfig) -> str:
    """Identity of what the run trains. Reporting-only sections and unset options are left out."""
    payload = cfg.model_dump(mode="json", exclude=HASH_EXCLUDED_SECTIONS)
    if payload.get("task") == "binary_classification":
        payload.pop("task", None)
    if payload.get("fitted_features") is None:
        payload.pop("fitted_features", None)
    if payload["split"].get("cv_seed") is None:
        payload["split"].pop("cv_seed", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def flatten_params(cfg: RunConfig) -> dict[str, str]:
    """Dotted key -> string value, for the run_params table."""
    flat: dict[str, str] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            flat[prefix] = json.dumps(node) if isinstance(node, list) else str(node)

    walk("", cfg.model_dump(mode="json"))
    return flat


def main(argv: list[str] | None = None) -> None:
    """Validate and print a config or a sweep. Runs nothing."""
    p = argparse.ArgumentParser(description="Validate a run config or sweep without running it.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--sweep", type=Path)
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    args = p.parse_args(argv)

    if args.sweep:
        configs = expand_sweep(args.sweep)
        print(f"{len(configs)} runs")
        for cfg in configs:
            print(config_hash(cfg), cfg.experiment_name)
    else:
        cfg = load_config(args.config, args.overrides)
        print(config_json(cfg))
        print("hash:", config_hash(cfg))


if __name__ == "__main__":
    main()
