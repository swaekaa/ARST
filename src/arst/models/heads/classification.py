"""
Classification head for ARST and baseline models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head with LayerNorm and GELU.

    Args:
        d_model: Input dimension (fused representation).
        num_classes: Number of output classes.
        d_hidden: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 10,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

        # Kaiming init for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] fused representation.

        Returns:
            logits: [B, num_classes]
        """
        return self.net(x)
