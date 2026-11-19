"Verified native adapters; no changes to legacy random-split training."

import importlib.metadata
import platform
import sqlite3
import tempfile
from pathlib import Path

from .learned_artifacts import file_hash, verify
from .temporal_compare import compare
from .temporal_samples import materialize, options
from .temporal_train import train


def producer():
    directory = Path(__file__).parent
    names = [
        "learned_artifacts.py",
        "temporal_samples.py",
        "temporal_train.py",
        "temporal_compare.py",
        "temporal_xgb_io.py",
        "temporal_xgb.py",
        "history.py",
        "history_contract.py",
        "history_io.py",
        "history_artifacts.py",
        "policy.py",
    ]
    hashes = {name: file_hash(directory / name) for name in names}
    hashes["models/xgb_regressor.py"] = file_hash(directory.parent / "models/xgb_regressor.py")
    return {
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "source_sha256": hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["xgboost", "numpy", "pandas", "pyarrow"]
        },
    }


def checked_context(config):
    from .history_io import context

    plan, silver = context(config["plan"], config["silver_dataset"])
    return (plan, silver)


def build_samples(config):
    from .history import HistoryIndex, recipe
    from .history_contract import RequestIndex
    from .history_io import records, requests

    plan, silver = checked_context(config)
    t = silver.schema.spec
    columns = [t["observation_id"], *t["series"], t["time"], t["target"]]
    if t.get("available_at"):
        columns.append(t["available_at"])
    end = max(w["refit_end"] for w in plan["identity"]["windows"])
    with tempfile.TemporaryDirectory(
        prefix="tsxgb-index-", dir=Path(config["output_root"]).parent
    ) as temporary:
        temporary = Path(temporary)
        index = RequestIndex(temporary / "requests.sqlite", plan, requests(config["plan"], plan))
        history = None
        try:
            history = HistoryIndex(
                temporary / "history.sqlite",
                plan["identity"]["policy"],
                t,
                plan["identity"]["holdout"]["start"],
                end,
            )
            history.add_rows(
                records(silver, columns, end, config["batch_size"]), config["batch_size"]
            )
            return materialize(
                plan,
                index,
                history,
                recipe(config["recipe"]),
                options(config["samples"]),
                producer(),
                config["output_root"],
                config["database"],
            )
        finally:
            if history is not None:
                history.close()
            index.close()


def require_cuda(booster):
    import json
    import re

    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise RuntimeError(
            f"CUDA requested but XGBoost resolved to {device!r}; refusing CPU fallback"
        )
    return device


def probe_cuda():
    import numpy as np
    import xgboost as xgb

    if xgb.__version__ != "2.1.1":
        raise RuntimeError("Temporal CUDA requires pinned XGBoost 2.1.1")
    x = np.arange(32, dtype=np.float32).reshape(16, 2)
    model = xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=1, max_depth=1)
    try:
        model.fit(x, x[:, 0])
        return require_cuda(model.get_booster())
    except (xgb.core.XGBoostError, RuntimeError) as exc:
        raise RuntimeError(
            "CUDA preflight failed; check nvidia-smi and the XGBoost build, or explicitly use cpu"
        ) from exc


def fit(config, samples):
    import xgboost

    from ml_template.models.xgb_regressor import XGBRegressionAdapter
    from ml_template.training.run_config import XGBRegressionParams

    from .history import recipe

    if xgboost.__version__ != "2.1.1":
        raise ValueError("This candidate targets the repository's pinned xgboost==2.1.1")
    plan, _ = checked_context(config)
    sample = verify(samples, "samples")["identity"]["inputs"]
    if (
        sample["plan_id"] != plan["plan_id"]
        or sample["plan_manifest_sha256"] != plan["manifest_sha256"]
        or sample["silver"] != plan["identity"]["silver"]
        or (sample["recipe"] != recipe(config["recipe"]))
        or (sample["sample_options"] != options(config["samples"]))
    ):
        raise ValueError("Sample config/plan/Silver binding mismatch")
    params = XGBRegressionParams.model_validate(config["model"])
    adapter = XGBRegressionAdapter
    if params.device == "cuda":
        probe_cuda()

        class CUDAAdapter(XGBRegressionAdapter):
            def fit_with_validation(self, X_train, y_train, X_val, y_val):
                rounds = super().fit_with_validation(X_train, y_train, X_val, y_val)
                self.effective_training_device = require_cuda(self.model.get_booster())
                return rounds

            def refit(self, X, y, n_rounds):
                super().refit(X, y, n_rounds)
                self.effective_training_device = require_cuda(self.model.get_booster())

        adapter = CUDAAdapter
    seed = config["seed"]
    return train(
        samples,
        params.model_dump(),
        seed,
        producer(),
        lambda: adapter(params, seed),
        lambda path: XGBRegressionAdapter.load(params, seed, path),
        config["output_root"],
        config["database"],
    )


def score(config, models, baseline_path):
    from .history_artifacts import verify_artifact
    from .history_contract import RequestIndex
    from .history_contract import producer as baseline_producer
    from .history_io import records, requests
    from .learned_artifacts import rows

    baseline = verify_artifact(baseline_path, "predictions")
    if baseline["identity"]["inputs"]["producer"] != baseline_producer():
        raise ValueError("Baseline producer changed; regenerate, do not reseal")
    plan, silver = checked_context(config)
    t = silver.schema.spec
    columns = [t["observation_id"], *t["series"], t["time"], t["target"]]
    low = min(w["target_date"] for w in plan["identity"]["windows"])
    high = max(w["target_date"] for w in plan["identity"]["windows"])
    if high >= plan["identity"]["holdout"]["start"]:
        raise ValueError("Holdout comparison prohibited")
    with tempfile.TemporaryDirectory(prefix="tsxgb-score-") as temporary:
        index = RequestIndex(
            Path(temporary) / "requests.sqlite", plan, requests(config["plan"], plan)
        )
        try:
            return compare(
                plan,
                index,
                models,
                baseline,
                rows(baseline_path, "predictions.jsonl"),
                records(silver, columns, high, config["batch_size"], low),
                producer(),
                config["output_root"],
                config["database"],
            )
        finally:
            index.close()
