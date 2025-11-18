# model.py
from __future__ import annotations

import torch
import torch.nn as nn


class MLPBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits (unnormalised scores)
        return self.net(x).squeeze(-1)


def build_model(input_dim: int) -> nn.Module:
    """Factory used by run.py / train.py.

    Agents are encouraged to modify this function (and the MLPBinaryClassifier)
    to explore better architectures.
    """
    return MLPBinaryClassifier(input_dim=input_dim)
