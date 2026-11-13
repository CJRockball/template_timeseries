"""Subprocess-isolated NN train/refit/checkpoint smoke; never release evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.model_selection import train_test_split

from ml_template.models.nn.params import NNModelConfig, NNParams


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["synthetic", "experiment"] = "synthetic"
    model_file: str | None = None
    experiment: str | None = None
    sample_rows: int = Field(128, ge=16)
    seed: int = Field(42, ge=0)
    validation_fraction: float = Field(0.25, gt=0, lt=0.5)
    timeout_seconds: int = Field(120, ge=1)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_source(self):
        if self.source == "synthetic" and (not self.model_file or self.experiment):
            raise ValueError("synthetic smoke requires model_file, not experiment")
        if self.source == "experiment" and (not self.experiment or self.model_file):
            raise ValueError("experiment smoke requires experiment, not model_file")
        return self


def read_spec(path):
    with Path(path).open(encoding="utf-8") as stream:
        return SmokeConfig.model_validate(yaml.safe_load(stream))


def _inner(adapter):
    while hasattr(adapter, "inner"):
        adapter = adapter.inner
    return adapter


def _inputs(spec):
    if spec.source == "synthetic":
        with Path(spec.model_file).open(encoding="utf-8") as stream:
            model = NNModelConfig.model_validate(yaml.safe_load(stream))
        params = NNParams.model_validate({**model.params.model_dump(), **spec.overrides})
        rng = np.random.default_rng(spec.seed)
        n = spec.sample_rows
        X = pd.DataFrame(
            {
                "num_0": rng.normal(size=n),
                "num_1": rng.normal(size=n),
                "cat_0": rng.choice(["A", "B", "C"], n),
            }
        )
        latent = X.num_0.to_numpy() + (X.cat_0 == "A").to_numpy() + rng.normal(size=n) * 0.2
        y = pd.Series((latent > np.median(latent)).astype(int), index=X.index)
        X.loc[rng.random(n) < 0.15, "num_0"] = np.nan
        X.loc[rng.random(n) < 0.15, "cat_0"] = None
        cfg = SimpleNamespace(
            task="binary_classification",
            model=NNModelConfig(params=params),
            seed=spec.seed,
            fitted_features=None,
        )
        return cfg, X, y
    from ml_template.data.contract import load_dataset_contract
    from ml_template.paths import DATA_SILVER, DATA_SPLITS
    from ml_template.training.run_config import load_config, resolve_path
    from ml_template.training.run_inputs import split_id_for

    cfg = load_config(spec.experiment)
    if cfg.model.type != "nn":
        raise ValueError("experiment must configure model.type: nn")
    if cfg.data.feature_set_id is not None:
        raise ValueError(
            "sampled experiment smoke requires feature_set_id: null; normal training supports FE"
        )
    params = NNParams.model_validate({**cfg.model.params.model_dump(), **spec.overrides})
    cfg = cfg.model_copy(update={"model": NNModelConfig(params=params), "seed": spec.seed})
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    split = split_id_for(cfg, contract)
    assignments = pd.read_parquet(
        DATA_SPLITS / split / "assignments.parquet", columns=[contract.entity_id, "partition"]
    )
    if (
        assignments[contract.entity_id].isna().any()
        or assignments[contract.entity_id].duplicated().any()
    ):
        raise ValueError("invalid assignment keys")
    dev = assignments.loc[assignments.partition.eq("dev"), contract.entity_id]
    keys = dev.sample(n=min(len(dev), spec.sample_rows), random_state=spec.seed).tolist()
    if len(keys) < 16:
        raise ValueError("insufficient development rows")
    path = DATA_SILVER / cfg.data.dataset_name / cfg.data.silver_dataset_version_id / "data.parquet"
    # Select development keys before any target labels are read.
    frame = pd.read_parquet(
        path,
        columns=[contract.entity_id, *contract.features, contract.target],
        filters=[(contract.entity_id, "in", keys)],
    ).set_index(contract.entity_id)
    if not frame.index.is_unique or set(frame.index) != set(keys):
        raise ValueError("sampled Silver keys differ from selected development keys")
    frame = frame.loc[keys]
    return cfg, frame.loc[:, list(contract.features)], frame[contract.target]


def run_smoke(spec, output_dir=None):
    from ml_template.models.model_factory import build_adapter_factory, load_adapter

    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError("smoke output directory already exists")
    cfg, X, y = _inputs(spec)
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("sample needs >= 2 examples of each class; increase sample_rows")
    fit, stop = train_test_split(
        np.arange(len(X)),
        test_size=spec.validation_fraction,
        random_state=spec.seed,
        stratify=y.to_numpy(),
    )
    make = build_adapter_factory(cfg)
    stopper = make()
    best = stopper.fit_with_validation(X.iloc[fit], y.iloc[fit], X.iloc[stop], y.iloc[stop])
    if not 1 <= best <= cfg.model.params.max_epochs:
        raise ValueError("invalid selected epoch")
    if next(_inner(stopper).model.parameters()).device.type != "cpu":
        raise ValueError("early-stop model was not offloaded to CPU")
    # Fresh refit on all sampled development rows, not a performance estimate.
    model = make()
    model.refit(X, y, best)
    before = _inner(model).preprocessor.state_dict()
    p = model.predict_proba(X)
    if p.shape != (len(X),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid NN probabilities")
    chunked = np.concatenate([model.predict_proba(X.iloc[:1]), model.predict_proba(X.iloc[1:])])
    np.testing.assert_allclose(p, chunked, rtol=0, atol=1e-6)
    np.testing.assert_allclose(p, model.predict_proba(X.iloc[::-1])[::-1], rtol=0, atol=1e-6)
    if before != _inner(model).preprocessor.state_dict():
        raise ValueError("prediction changed fitted preprocessing")
    with tempfile.TemporaryDirectory(prefix="nn_training_smoke_") as temp:
        path = Path(temp) / "model.ubj"
        model.save(path)
        restored = load_adapter(cfg, path)
        np.testing.assert_allclose(p, restored.predict_proba(X), rtol=0, atol=1e-6)
    report = {
        "status": "passed",
        "scope": "training smoke, not performance/release evidence",
        "n_rows": len(X),
        "n_fit": len(fit),
        "n_stop": len(stop),
        "best_epoch": best,
        "training_device": cfg.model.params.device,
        "inference_device": cfg.model.params.inference_device,
        "source": spec.source,
        "holdout_labels_loaded": False,
        "normal_run_registered": False,
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "checks": [
            "finite_backward",
            "inner_validation",
            "fresh_refit",
            "checkpoint_reload",
            "chunk_shuffle_singleton_parity",
            "frozen_preprocessing",
            "cpu_offload",
        ],
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        model.save(destination / "model.ubj")
        (destination / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "resolved_params.json").write_text(
            cfg.model.params.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec = read_spec(args.config)
    if args.worker:
        print(json.dumps(run_smoke(spec, args.output_dir), indent=2))
        return
    command = [
        sys.executable,
        "-m",
        "ml_template.training.nn_smoke",
        "--config",
        str(args.config),
        "--worker",
    ]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    try:
        result = subprocess.run(command, timeout=spec.timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("NN smoke exceeded timeout; child process terminated") from exc
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
