"""Embedding MLP with optional numeric/categorical branches and one logit."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingMLP(nn.Module):
    def __init__(self, n_numeric, cardinalities, params):
        super().__init__()
        if n_numeric < 0 or n_numeric + len(cardinalities) < 1:
            raise ValueError("MLP requires at least one input branch")
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, min(params.embedding_max_dim, (size + 1) // 2))
                for size in cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -0.01, 0.01)
            with torch.no_grad():
                embedding.weight[:2].zero_()
        self.embedding_dropout = nn.Dropout(params.embedding_dropout)
        numeric_dim = n_numeric * params.numeric_projection_multiplier
        self.numeric_projection = (
            nn.Sequential(nn.Linear(n_numeric, numeric_dim), nn.ReLU()) if n_numeric else None
        )
        width = numeric_dim + sum(e.embedding_dim for e in self.embeddings)
        layers = []
        for output in params.hidden_dims:
            layers.extend([nn.Linear(width, output), nn.SiLU()])
            if params.normalization == "layernorm":
                layers.append(nn.LayerNorm(output))
            layers.append(nn.Dropout(params.dropout))
            width = output
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, numeric, categorical):
        parts = []
        if self.embeddings:
            embedded = torch.cat(
                [e(categorical[:, i]) for i, e in enumerate(self.embeddings)], dim=1
            )
            parts.append(self.embedding_dropout(embedded))
        if self.numeric_projection is not None:
            parts.append(self.numeric_projection(numeric))
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(combined).squeeze(-1)

    @torch.no_grad()
    def refresh_unknown_embeddings(self):
        for embedding in self.embeddings:
            if embedding.num_embeddings > 2:
                embedding.weight[1].copy_(embedding.weight[2:].mean(dim=0))
            else:
                embedding.weight[1].zero_()
