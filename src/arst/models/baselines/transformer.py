"""
Transformer Baseline for ARST.

Concatenates all modalities along the feature axis and applies a standard
Transformer encoder for temporal modeling. No reliability module, no
adaptive fusion — serves as the strongest non-ARST baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arst.models.encoders.imu_encoder import SinusoidalPositionalEncoding


class TransformerBaseline(nn.Module):
    """
    Transformer baseline: project concatenated modalities → Transformer → MLP head.

    Args:
        in_channels_total: Total input channels (6 + 64 + 64 = 134).
        d_model: Transformer embedding dimension.
        num_layers: Transformer encoder depth.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        num_classes: Number of behavior classes.
        pool_type: "cls" | "mean".
    """

    def __init__(
        self,
        in_channels_total: int = 134,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 10,
        pool_type: str = "cls",
    ):
        super().__init__()
        self.pool_type = pool_type

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels_total, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor,
        tof: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            imu:    [B, T, 6]
            thermo: [B, T, 64]
            tof:    [B, T, 64]

        Returns:
            logits: [B, num_classes]
        """
        B, T, _ = imu.shape

        # Concatenate modalities along feature axis
        x = torch.cat([imu, thermo, tof], dim=-1)  # [B, T, 134]

        # Project to d_model
        x = self.input_proj(x)  # [B, T, d_model]

        # CLS token
        if self.pool_type == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, 1+T, d_model]

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Pool
        if self.pool_type == "cls":
            pooled = x[:, 0, :]
        else:
            pooled = x.mean(dim=1)

        return self.head(pooled)
