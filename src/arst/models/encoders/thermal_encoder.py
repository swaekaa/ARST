"""
Thermal Encoder: Linear projection + Transformer for thermopile array encoding.

Handles 8×8 (64-channel) thermopile data as a flattened spatial vector
processed over the temporal dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arst.models.encoders.imu_encoder import SinusoidalPositionalEncoding


class ThermalEncoder(nn.Module):
    """
    Thermal Encoder for 8×8 thermopile array data.

    Architecture (linear_transformer variant):
        Input [B, T, 64]
        → Linear(64, d_model)
        → LayerNorm
        → Positional Encoding
        → Transformer Encoder (L layers)
        → Output [B, T, d_model]

    Args:
        in_channels: Input feature dim (64 for 8×8 array).
        d_model: Output embedding dimension.
        num_transformer_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        pos_encoding: "sinusoidal" | "learned".
    """

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        else:
            from arst.models.encoders.imu_encoder import LearnedPositionalEncoding

            self.pos_enc = LearnedPositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 64] thermopile sequence.

        Returns:
            H: [B, T, d_model] encoded representation.
        """
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)
        H = self.transformer(x)
        return H
