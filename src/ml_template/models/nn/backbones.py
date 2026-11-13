"""Architecture extension point; torch is imported only during construction."""


def build_backbone(n_numeric, cardinalities, params):
    if params.architecture != "mlp":
        raise ValueError("only MLP is currently implemented")
    import torch

    from ml_template.models.nn.mlp import EmbeddingMLP

    with torch.device("cpu"):
        return EmbeddingMLP(n_numeric, cardinalities, params).float()
