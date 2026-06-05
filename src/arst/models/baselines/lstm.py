"""
BiLSTM Baseline for ARST Phase 2.

Architecture (per modality, then fused):
    Input [B, T, C_m]
        → Linear projection (C_m → lstm_input_dim)
        → BiLSTM(hidden_size, num_layers, bidirectional=True)
        → Self-attention pooling over time → [B, 2*hidden_size]
    Concatenate active modality features
        → Linear → GELU → Dropout
        → Linear → logits [B, num_classes]

Design notes:
    - Phase 1 dims: IMU=7, Thermal=5, ToF=320.
    - BiLSTM captures temporal dependencies more effectively than MLP.
    - Attention pooling (learned query) outperforms naive last-state or
      mean-pool for variable-length sequences.
    - ToF: projected to a compact ``lstm_input_dim`` before LSTM to avoid
      O(320 × hidden) weight matrices.
    - ``active_modalities`` controls which branches are active.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMU_CH = 7
_THM_CH = 5
_TOF_CH = 320


class AttentionPool(nn.Module):
    """
    Single-query self-attention temporal pooling.

    Learns a query vector that attends over all timesteps and returns a
    weighted sum — more expressive than mean or last-step pooling.

    Args:
        d_in: Feature dimension of LSTM output.
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_in, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_in]

        Returns:
            pooled: [B, d_in]
        """
        scores = self.query(x).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)  # [B, T]
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, d_in]


class _LSTMBranch(nn.Module):
    """Single-modality BiLSTM branch."""

    def __init__(
        self,
        in_channels: int,
        lstm_input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, lstm_input_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden_size * 2)
        self.out_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = self.proj(x)  # [B, T, lstm_input_dim]
        out, _ = self.lstm(x)  # [B, T, 2*hidden]
        return self.pool(out)  # [B, 2*hidden]


class LSTMBaseline(nn.Module):
    """
    Bidirectional LSTM Baseline with per-modality branches.

    Args:
        num_classes:       Number of behavior classes (4).
        imu_channels:      IMU channels (7).
        thermal_channels:  Thermopile channels (5).
        tof_channels:      ToF channels (320).
        lstm_input_dim:    Projection dim before LSTM (all modalities).
        hidden_size:       LSTM hidden size per direction.
        num_layers:        Stacked LSTM depth.
        head_hidden_dim:   Classification head hidden dim.
        dropout:           Dropout in LSTM and head.
        active_modalities: Which modalities to include.
    """

    def __init__(
        self,
        num_classes: int = 4,
        imu_channels: int = _IMU_CH,
        thermal_channels: int = _THM_CH,
        tof_channels: int = _TOF_CH,
        lstm_input_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        active_modalities: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.active_modalities: list[str] = (
            active_modalities if active_modalities is not None else ["imu", "thermo", "tof"]
        )
        fused_dim = 0

        if "imu" in self.active_modalities:
            self.imu_branch = _LSTMBranch(
                imu_channels, lstm_input_dim, hidden_size, num_layers, dropout
            )
            fused_dim += self.imu_branch.out_dim

        if "thermo" in self.active_modalities:
            self.thermo_branch = _LSTMBranch(
                thermal_channels, lstm_input_dim, hidden_size, num_layers, dropout
            )
            fused_dim += self.thermo_branch.out_dim

        if "tof" in self.active_modalities:
            self.tof_branch = _LSTMBranch(
                tof_channels, lstm_input_dim, hidden_size, num_layers, dropout
            )
            fused_dim += self.tof_branch.out_dim

        if fused_dim == 0:
            raise ValueError("At least one modality must be active.")

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

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
            tof_mask: Accepted but ignored (Phase 2 — no reliability).

        Returns:
            logits: [B, num_classes]
        """
        parts: list[torch.Tensor] = []

        if "imu" in self.active_modalities:
            assert imu is not None
            parts.append(self.imu_branch(imu))

        if "thermo" in self.active_modalities:
            assert thermo is not None
            parts.append(self.thermo_branch(thermo))

        if "tof" in self.active_modalities:
            assert tof is not None
            parts.append(self.tof_branch(tof))

        x = torch.cat(parts, dim=-1)
        return self.head(x)
