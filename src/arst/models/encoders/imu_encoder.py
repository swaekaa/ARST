"""
IMU Encoder: Multi-scale CNN + Transformer for motion signal encoding.

Architecture:
  Input [B, T, 6]
  → Multi-scale 1D Convolution (parallel branches: kernel 3, 7, 15)
  → Feature concatenation + projection to d_model
  → Positional encoding
  → Stacked Transformer encoder blocks
  → Output [B, T, d_model]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange


class MultiScaleConvBlock(nn.Module):
    """Parallel 1D convolutions at multiple temporal scales."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int]):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels // len(kernel_sizes),
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return out  # [B, out_channels, T]


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class IMUEncoder(nn.Module):
    """
    IMU Encoder: Multi-scale temporal convolution followed by Transformer encoding.

    Args:
        in_channels: Number of input channels (default 6: acc + gyro).
        d_model: Output embedding dimension.
        cnn_filters: Number of filters per CNN branch.
        cnn_kernel_sizes: Kernel sizes for multi-scale convolution.
        cnn_dropout: Dropout after CNN.
        num_transformer_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dimension in Transformer.
        transformer_dropout: Dropout in Transformer.
        pos_encoding: "sinusoidal" | "learned".
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 256,
        cnn_filters: int = 64,
        cnn_kernel_sizes: list[int] = [3, 7, 15],
        cnn_dropout: float = 0.1,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        transformer_dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_channels, cnn_filters)

        # Multi-scale CNN
        self.multi_scale_conv = MultiScaleConvBlock(
            in_channels=cnn_filters,
            out_channels=cnn_filters * len(cnn_kernel_sizes),
            kernel_sizes=cnn_kernel_sizes,
        )
        cnn_out_channels = cnn_filters * len(cnn_kernel_sizes)

        # Project to d_model
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=transformer_dropout)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, dropout=transformer_dropout)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.dropout = nn.Dropout(cnn_dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 6] IMU sequence.

        Returns:
            H: [B, T, d_model] encoded representation.
        """
        B, T, C = x.shape

        # Input projection
        x = self.input_proj(x)  # [B, T, cnn_filters]

        # CNN expects [B, C, T]
        x = rearrange(x, "b t c -> b c t")
        x = self.multi_scale_conv(x)  # [B, cnn_out_channels, T]
        x = rearrange(x, "b c t -> b t c")

        # Project to d_model
        x = self.proj(x)  # [B, T, d_model]
        x = self.dropout(x)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoding
        H = self.transformer(x)  # [B, T, d_model]

        return H


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pe(positions).unsqueeze(0)
        return self.dropout(x)
