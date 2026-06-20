"""
Transformer Baseline for ARST Phase 2.

Architecture:
    Input: [B, T, C_imu] + [B, T, C_thm] + [B, T, C_tof]
        → Per-modality linear projection to d_model
        → Concatenate along time: [B, T_total, d_model]
              (T_total = T per active modality)
        → Sinusoidal positional encoding
        → Transformer encoder (L layers, H heads, d_ff FFN)
        → [CLS] token or mean pooling
        → Linear → GELU → Dropout
        → Linear → logits [B, num_classes]

This is the **strongest non-ARST baseline** — it has cross-modal
attention but no per-timestep reliability gating.

Design notes:
    - Phase 1 dims: IMU=7, Thermal=5, ToF=320.
    - ToF 320 channels projected to d_model via Linear; no spatial CNN.
    - CLS token is prepended per-modality (not global) so the model
      learns a per-modality summary before cross-modal attention.
    - Reduced defaults (d_model=128, L=2, H=4) to fit RTX 3060 4 GB
      in Phase 2. Phase 3 encoders will use larger Transformers.
    - ``active_modalities`` enables unimodal ablation variants.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Reuse from existing encoder (already in codebase)
from arst.models.encoders.imu_encoder import SinusoidalPositionalEncoding

_IMU_CH = 7
_THM_CH = 5
_TOF_CH = 320


class TransformerBaseline(nn.Module):
    """
    Transformer Baseline: modality projection + shared Transformer encoder.

    All active modalities are projected to ``d_model``, concatenated along
    the time axis, then processed by a single shared Transformer encoder.
    A prepended [CLS] token provides a global sequence representation.

    Args:
        num_classes:       Number of behavior classes (4).
        imu_channels:      IMU channels (7).
        thermal_channels:  Thermopile channels (5).
        tof_channels:      ToF channels (320).
        d_model:           Transformer embedding dimension.
        num_layers:        Transformer encoder depth.
        num_heads:         Number of attention heads (must divide d_model).
        d_ff:              Feed-forward dimension inside Transformer blocks.
        dropout:           Dropout probability.
        pool_type:         ``"cls"`` (CLS token, default) or ``"mean"``.
        active_modalities: Which modalities to include in the sequence.
    """

    def __init__(
        self,
        num_classes: int = 4,
        imu_channels: int = _IMU_CH,
        thermal_channels: int = _THM_CH,
        tof_channels: int = _TOF_CH,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pool_type: str = "cls",
        active_modalities: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.active_modalities: list[str] = (
            active_modalities if active_modalities is not None else ["imu", "thermo", "tof"]
        )
        self.pool_type = pool_type
        self.d_model = d_model

        if not self.active_modalities:
            raise ValueError("At least one modality must be active.")

        # Per-modality input projections
        # NOTE: These use default Kaiming init — _init_weights() must NOT
        # override them with trunc_normal_ (that was the Phase 2.5 bug).
        if "imu" in self.active_modalities:
            self.imu_proj = nn.Linear(imu_channels, d_model)

        if "thermo" in self.active_modalities:
            self.thermo_proj = nn.Linear(thermal_channels, d_model)

        if "tof" in self.active_modalities:
            self.tof_proj = nn.Linear(tof_channels, d_model)

        # Shared input norm — stabilises projected features before Transformer
        self.input_norm = nn.LayerNorm(d_model)

        # Embedding scale factor (Vaswani et al., "Attention Is All You Need")
        self._embed_scale = math.sqrt(d_model)

        # CLS token (single global token prepended to entire concatenated sequence)
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding — sinusoidal, shared across modalities
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Shared Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise Transformer + head layers with truncated normal (ViT-style).

        IMPORTANT: Input projection layers (imu_proj, thermo_proj, tof_proj)
        are intentionally EXCLUDED — they must keep the default Kaiming
        initialisation so they can project raw sensor values (which have
        much larger magnitudes than d_model-scale embeddings).  Overwriting
        them with trunc_normal_(std=0.02) was the root cause of the Phase 2.5
        Transformer collapse (F1 = 0.035).
        """
        # Collect input projection layers that must be skipped
        skip_modules = set()
        for name in ("imu_proj", "thermo_proj", "tof_proj"):
            if hasattr(self, name):
                skip_modules.add(id(getattr(self, name)))

        for module in self.modules():
            if id(module) in skip_modules:
                continue  # keep Kaiming init for input projections
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor | None = None,
        tof: torch.Tensor | None = None,
        tof_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            imu:      [B, T, 7]    IMU sequence.
            thermo:   [B, T, 5]    Thermopile sequence (None if inactive).
            tof:      [B, T, 320]  ToF sequence (None if inactive).
            tof_mask: Accepted but ignored (Phase 2 — no reliability module).

        Returns:
            logits: [B, num_classes]
        """
        B = (imu if imu is not None else thermo if thermo is not None else tof).shape[0]

        segments: list[torch.Tensor] = []

        if "imu" in self.active_modalities:
            assert imu is not None
            segments.append(self.imu_proj(imu))  # [B, T, d_model]

        if "thermo" in self.active_modalities:
            assert thermo is not None
            segments.append(self.thermo_proj(thermo))  # [B, T, d_model]

        if "tof" in self.active_modalities:
            assert tof is not None
            segments.append(self.tof_proj(tof))  # [B, T, d_model]

        # Concatenate all modality tokens along time axis
        x = torch.cat(segments, dim=1)  # [B, T*n_modalities, d_model]

        # Normalise + scale projected features (Vaswani et al.)
        x = self.input_norm(x) * self._embed_scale

        # Prepend CLS token
        if self.pool_type == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls, x], dim=1)  # [B, 1 + T*n_modalities, d_model]

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Pooling
        if self.pool_type == "cls":
            pooled = x[:, 0, :]  # [B, d_model]
        else:
            pooled = x.mean(dim=1)  # [B, d_model]

        return self.head(pooled)
