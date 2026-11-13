"""Build feature blocks and register a feature set.

    python -m ml_template.features.build_features --config configs/experiment/xgb_baseline.yaml \\
        --blocks thermal_power strain

Prints the feature set ID to put in ``data.feature_set_id``. With ``--parent`` the new set is
the parent plus the given blocks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from ml_template.data.contract import load_dataset_contract
from ml_template.data.silver_loader import load_silver
from ml_template.features.block_registry import available_blocks, discover_blocks
from ml_template.features.feature_set_store import (
    build_block,
    create_feature_set,
    extend_feature_set,
    load_feature_set,
    register_block_record,
    register_feature_set_record,
)
from ml_template.paths import DATA_FEATURES, DATA_SILVER
from ml_template.training.run_config import load_config, resolve_path

logger = logging.getLogger("ml_template.features.build_features")


def main(argv: list[str] | None = None) -> str:
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    discover_blocks()
    p = argparse.ArgumentParser(description="Build feature blocks and a feature set.")
    p.add_argument("--config", default="configs/experiment/xgb_baseline.yaml", type=Path)
    p.add_argument("--blocks", nargs="+", required=True, help=f"one of {available_blocks()}")
    p.add_argument("--parent", default=None, help="feature set to extend")
    p.add_argument("--description", default="")
    p.add_argument("--silver-version-id", default=None, help="Overrides the config value")
    p.add_argument("--silver-root", default=DATA_SILVER, type=Path)
    p.add_argument("--features-root", default=DATA_FEATURES, type=Path)
    p.add_argument("--no-register", action="store_true", help="skip the database rows")
    p.add_argument(
        "--block-params", type=Path, default=None, help="YAML mapping block name to its parameters"
    )
    p.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Final model columns to keep, in order; default keeps all available columns",
    )
    args = p.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    silver_id = args.silver_version_id or cfg.data.silver_dataset_version_id
    silver = load_silver(
        silver_id,
        cfg.data.dataset_name,
        contract.entity_id,
        args.silver_root,
        task=cfg.task,
    )
    raw = silver.model_frame()[silver.features]

    block_params = {}
    if args.block_params:
        block_params = yaml.safe_load(args.block_params.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(block_params) - set(args.blocks))
    if unknown:
        raise ValueError(f"--block-params has blocks not in --blocks: {unknown}")
    manifests = [
        build_block(b, raw, silver_id, params=block_params.get(b), root=args.features_root)
        for b in args.blocks
    ]

    block_ids = [m["block_id"] for m in manifests]

    if args.columns is not None:
        # Include the parent's blocks so selection can use its existing features,
        # even when --blocks lists only newly added blocks.
        selected_block_ids = list(block_ids)

        if args.parent is not None:
            parent_set = load_feature_set(args.parent, args.features_root)

            if parent_set["silver_dataset_version_id"] != silver_id:
                raise ValueError(
                    f"parent {args.parent} belongs to "
                    f"{parent_set['silver_dataset_version_id']}, not {silver_id}"
                )

            selected_block_ids = list(dict.fromkeys([*parent_set["blocks"], *block_ids]))

        fs = create_feature_set(
            silver_id,
            selected_block_ids,
            raw_columns=silver.features,
            columns=args.columns,
            parent=args.parent,
            description=args.description,
            root=args.features_root,
        )

    elif args.parent is not None:
        fs = extend_feature_set(
            args.parent,
            block_ids,
            description=args.description,
            root=args.features_root,
        )

    else:
        fs = create_feature_set(
            silver_id,
            block_ids,
            raw_columns=silver.features,
            description=args.description,
            root=args.features_root,
        )

    if not args.no_register:
        engine = get_engine()
        for manifest in manifests:
            register_block_record(engine, manifest)
        register_feature_set_record(engine, fs, args.features_root)
    logger.info("Feature set %s: %d features", fs["feature_set_id"], fs["n_features"])
    print(fs["feature_set_id"])
    return fs["feature_set_id"]


if __name__ == "__main__":
    main()
