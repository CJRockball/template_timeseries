"""Release schemas for model-neutral, nullable features BEFORE fitted Gold steps."""

from ml_template.scoring.input_contract import frame_schema


def build_pre_gold_contract(
    source,
    preprocessor_inputs,
    *,
    key_col,
    key_kind,
    input_spec,
    source_policy=None,
    model_policy=None,
    legacy_model_limits=None,
):
    """Freeze feature structure, vocabulary and observed null rates, not imputation.

    Source/model policies are retained as historical reference, not used as null
    compatibility gates in this V3 contract. IDs remain strict in validate_batch.
    """
    if legacy_model_limits:
        raise ValueError(
            "V3 uses nullable pre-Gold features. Remove model_max_missing_rate "
            "from this release request; rate limits are separate data-quality policy."
        )
    if key_kind not in {"int", "str"}:
        raise ValueError("unsupported key kind")
    if key_col in source.columns or key_col in preprocessor_inputs.columns:
        raise ValueError("key must be the index, not a model feature")
    preprocessor_schema = frame_schema(preprocessor_inputs, nullable=True)
    return {
        "schema_version": 3,
        "key_col": key_col,
        "key_kind": key_kind,
        "source_schema": frame_schema(source, nullable=True),
        "preprocessor_input_schema": preprocessor_schema,
        # Compatibility alias: this has NEVER represented post-fitted-transform inputs.
        "model_schema": preprocessor_schema,
        "model_schema_stage": "pre_gold",
        "input_spec": input_spec,
        "missing_policy": "preserve_pre_gold",
        "missing_rate_enforcement": "not_in_this_scoring_contract",
        "legacy_missingness_reference": {
            "source": source_policy or {},
            "preprocessor": model_policy or {},
        },
        "gold_preprocessing": "owned_by_saved_model_adapter",
        "estimator_validation": "adapter_owned; finite NN contract not yet wired",
        "unseen_category_policy": "error",
        "extra_columns": "error",
        "vocabulary_origin": "feature-only replay of saved training preparation; no target read",
    }


def get_preprocessor_schema(contract):
    version = contract.get("schema_version")
    if version not in {1, 2, 3}:
        raise ValueError("unsupported scoring contract")
    if version == 3:
        schema = contract["preprocessor_input_schema"]
        if "model_schema" in contract and contract["model_schema"] != schema:
            raise ValueError("legacy schema alias differs from preprocessor schema")
        return schema
    return contract["model_schema"]
