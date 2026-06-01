"""
ToF Encoder: Handles time-of-flight depth array with learned null embedding for invalid pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arst.models.encoders.imu_encoder import SinusoidalPositionalEncoding


class ToFEncoder(nn.Module):
    """
    ToF Encoder for 8×8 time-of-flight depth array.

    Key feature: invalid ToF readings (flagged by tof_mask=0) are replaced
    with a learned null embedding rather than zero, allowing the model to
    distinguish between "zero depth" and "invalid reading".

    Architecture:
        Input [B, T, 64], Mask [B, T, 64]
        → Null embedding replacement for masked pixels
        → Linear(64, d_model)
        → LayerNorm
        → Positional Encoding
        → Transformer Encoder
        → Output [B, T, d_model]

    Args:
        in_channels: Input feature dim (64 for 8×8 array).
        d_model: Output embedding dimension.
        use_invalid_embedding: If True, use learned null embedding for masked pixels.
        num_transformer_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 256,
        use_invalid_embedding: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
    ):
        super().__init__()
        self.use_invalid_embedding = use_invalid_embedding
        self.in_channels = in_channels

        # Learned null value for invalid pixels (replaces 0)
        if use_invalid_embedding:
            self.null_pixel_value = nn.Parameter(torch.zeros(in_channels))

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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, T, 64] ToF depth array (already normalized; invalid=0).
            mask: [B, T, 64] binary mask (1=valid pixel, 0=invalid).

        Returns:
            H: [B, T, d_model] encoded representation.
        """
        if mask is not None and self.use_invalid_embedding:
            # Replace invalid pixels with learned null embedding
            # null_pixel_value: [64] → broadcast to [B, T, 64]
            null = self.null_pixel_value.unsqueeze(0).unsqueeze(0)
            x = x * mask + null * (1.0 - mask)

        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)
        H = self.transformer(x)
        return H
