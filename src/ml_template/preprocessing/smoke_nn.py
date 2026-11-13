"""CPU-only preprocessing smoke check; no model training, holdout or run registry.

python -m ml_template.preprocessing.smoke_nn --config configs/smoke/nn_preprocessing.yaml
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml_template.preprocessing.nn_config import load_nn_preprocessor_config
from ml_template.preprocessing.nn_preprocessor import NNPreprocessor


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _source(raw, config, rng):
    if not isinstance(raw, dict) or raw.get("kind") not in ("synthetic", "development"):
        raise ValueError("source.kind must be synthetic or development")
    columns = list(config.input_columns)
    if raw["kind"] == "synthetic":
        if set(raw) - {"kind", "rows"}:
            raise ValueError("unknown synthetic source settings")
        rows = _positive_integer(raw.get("rows", 128), "source.rows", 4)
        data = {}
        for column in config.numeric:
            values = rng.normal(size=rows)
            values[rng.random(rows) < 0.2] = np.nan
            data[column] = values
        for column in config.categorical:
            values = rng.choice(np.array(["A", "B", "C", None], dtype=object), size=rows)
            data[column] = values
        return pd.DataFrame(data)
    if set(raw) != {"kind", "data", "assignments", "key"}:
        raise ValueError("development source requires data, assignments and key")
    key = raw["key"]
    if not isinstance(key, str) or key in columns:
        raise ValueError("source key must be a non-feature string column")
    assignments = pd.read_parquet(raw["assignments"], columns=[key, "partition"])
    if assignments[key].isna().any() or assignments[key].duplicated().any():
        raise ValueError("invalid assignment keys")
    keys = assignments.loc[assignments.partition.eq("dev"), key]
    if keys.empty:
        raise ValueError("no development rows in assignments")
    path = Path(raw["data"])
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, usecols=[key, *columns])
    elif path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path, columns=[key, *columns])
    else:
        raise ValueError("development source must be CSV or Parquet")
    if data[key].isna().any() or data[key].duplicated().any():
        raise ValueError("invalid source keys")
    frame = data.set_index(key)
    if not set(keys) <= set(frame.index):
        raise ValueError("development keys missing from source (check key dtypes)")
    return frame.loc[keys, columns]


def run_smoke(path: str | Path, output_dir: str | Path | None = None) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) - {
        "preprocessor",
        "source",
        "sample_rows",
        "seed",
        "validation_fraction",
    }:
        raise ValueError("invalid smoke configuration")
    seed = _positive_integer(raw.get("seed", 42), "seed", 0)
    sample_rows = _positive_integer(raw.get("sample_rows", 128), "sample_rows", 4)
    fraction = raw.get("validation_fraction", 0.25)
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    config = load_nn_preprocessor_config(raw["preprocessor"])
    rng = np.random.default_rng(seed)
    source = _source(raw["source"], config, rng)
    positions = rng.permutation(len(source))[:sample_rows]
    if len(positions) < 4:
        raise ValueError("smoke test requires at least four development/synthetic rows")
    sample = source.iloc[positions]
    n_valid = max(1, min(len(sample) - 1, round(len(sample) * fraction)))
    train, valid = sample.iloc[n_valid:], sample.iloc[:n_valid]
    pre = NNPreprocessor(config).fit(train)
    before = pre.state_dict()
    train_arrays = pre.transform_arrays(train)
    expected = pre.transform(valid)
    arrays = pre.transform_arrays(valid)
    pd.testing.assert_frame_equal(
        expected, pd.concat([pre.transform(valid.iloc[:1]), pre.transform(valid.iloc[1:])])
    )
    reverse = pre.transform(valid.iloc[::-1]).iloc[::-1]
    pd.testing.assert_frame_equal(expected, reverse)
    pd.testing.assert_frame_equal(expected.iloc[:1], pre.transform(valid.iloc[:1]))
    if pre.state_dict() != before:
        raise ValueError("inference mutated fitted preprocessing state")
    with tempfile.TemporaryDirectory(prefix="nn_preprocessor_smoke_") as temporary:
        state_path = Path(temporary) / "preprocessor.json"
        pre.save(state_path)
        loaded = NNPreprocessor.load(state_path)
        pd.testing.assert_frame_equal(expected, loaded.transform(valid))
    report = {
        "status": "passed",
        "scope": "preprocessing_only",
        "device": "cpu",
        "source_kind": raw["source"]["kind"],
        "holdout_labels_loaded": False,
        "model_trained": False,
        "n_fit": len(train),
        "n_validation": len(valid),
        "numeric_shape": list(arrays.numeric.shape),
        "categorical_shape": list(arrays.categorical.shape),
        "numeric_names": list(arrays.numeric_names),
        "categorical_names": list(arrays.categorical_names),
        "cardinalities": list(arrays.cardinalities),
        "all_missing_fit_columns": [
            c for c, s in before["numeric_state"].items() if s["n_observed"] == 0
        ],
        "fit_numeric_shape": list(train_arrays.numeric.shape),
        "checks": [
            "finite_arrays",
            "embedding_bounds",
            "fixed_schema",
            "frozen_state",
            "save_load_parity",
            "chunk_parity",
            "singleton_parity",
            "shuffle_parity",
        ],
        "sample_positions": positions.tolist(),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        pre.save(destination / "preprocessor.json")
        (destination / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "config.json").write_text(
            json.dumps(raw, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
