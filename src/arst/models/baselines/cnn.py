"""
CNN Baseline for ARST Phase 2.

Architecture (per modality, then fused):
    Input [B, T, C_m]
        → Conv1D(k=3) + BN + GELU
        → Conv1D(k=7) + BN + GELU
        → Global Average Pool → [B, out_channels]
    Concatenate active modality features
        → Linear → GELU → Dropout
        → Linear → logits [B, num_classes]

Design notes:
    - Phase 1 dims: IMU=7, Thermal=5, ToF=320.
    - Per-modality 1D CNN branches are processed independently then concatenated.
    - ToF: 320 channels are projected to ``tof_proj_dim`` first to reduce
      parameter count before convolution (otherwise the conv layer alone
      would have 320 × 64 × 7 = 143,360 params per kernel).
    - ``active_modalities`` enables unimodal and multimodal ablations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_IMU_CH = 7
_THM_CH = 5
_TOF_CH = 320


def _cnn_branch(
    in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...]
) -> nn.Sequential:
    """Build a single-modality 1D-CNN branch."""
    layers: list[nn.Module] = []
    ch = in_channels
    for k in kernel_sizes:
        layers.extend(
            [
                nn.Conv1d(ch, out_channels, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ]
        )
        ch = out_channels
    # Global average pool collapses time dimension
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


class CNNBaseline(nn.Module):
    """
    1D-CNN Baseline: per-modality temporal convolution + classifier.

    Each active modality has its own independent CNN branch.  The branch
    outputs (global average pooled) are concatenated and fed to an MLP
    classification head.

    Args:
        num_classes:       Number of behavior classes (4).
        imu_channels:      IMU channels (7).
        thermal_channels:  Thermopile channels (5).
        tof_channels:      ToF channels (320).
        cnn_out_channels:  Output channels per CNN branch.
        kernel_sizes:      Kernel sizes for the two conv layers.
        tof_proj_dim:      Project ToF 320→``tof_proj_dim`` before CNN
                           to control parameter count.
        head_hidden_dim:   Hidden size of the classification head.
        dropout:           Dropout in the classification head.
        active_modalities: Which modalities to use (see :class:`MLPBaseline`).
    """

    def __init__(
        self,
        num_classes: int = 4,
        imu_channels: int = _IMU_CH,
        thermal_channels: int = _THM_CH,
        tof_channels: int = _TOF_CH,
        cnn_out_channels: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 7),
        tof_proj_dim: int = 64,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        active_modalities: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.active_modalities: list[str] = (
            active_modalities if active_modalities is not None else ["imu", "thermo", "tof"]
        )
        fused_dim = 0

        # IMU branch
        if "imu" in self.active_modalities:
            self.imu_cnn = _cnn_branch(imu_channels, cnn_out_channels, kernel_sizes)
            fused_dim += cnn_out_channels

        # Thermal branch
        if "thermo" in self.active_modalities:
            self.thermo_cnn = _cnn_branch(thermal_channels, cnn_out_channels, kernel_sizes)
            fused_dim += cnn_out_channels

        # ToF branch — project first to reduce param count
        if "tof" in self.active_modalities:
            self.tof_proj = nn.Linear(tof_channels, tof_proj_dim)
            self.tof_cnn = _cnn_branch(tof_proj_dim, cnn_out_channels, kernel_sizes)
            fused_dim += cnn_out_channels

        if fused_dim == 0:
            raise ValueError("At least one modality must be active.")

        # Classification head
        self.head = nn.Sequential(
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
            thermo:   [B, T, 5]    Thermopile sequence (or None if inactive).
            tof:      [B, T, 320]  ToF sequence (or None if inactive).
            tof_mask: Accepted but ignored (Phase 2 — no reliability module).

        Returns:
            logits: [B, num_classes]
        """
        parts: list[torch.Tensor] = []

        if "imu" in self.active_modalities:
            assert imu is not None
            # [B, T, 7] → [B, 7, T] → branch → [B, out_ch, 1] → [B, out_ch]
            feat = self.imu_cnn(imu.permute(0, 2, 1)).squeeze(-1)
            parts.append(feat)

        if "thermo" in self.active_modalities:
            assert thermo is not None
            feat = self.thermo_cnn(thermo.permute(0, 2, 1)).squeeze(-1)
            parts.append(feat)

        if "tof" in self.active_modalities:
            assert tof is not None
            # Project 320 → tof_proj_dim along feature axis first
            tof_p = self.tof_proj(tof)  # [B, T, tof_proj_dim]
            feat = self.tof_cnn(tof_p.permute(0, 2, 1)).squeeze(-1)
            parts.append(feat)

        x = torch.cat(parts, dim=-1)  # [B, fused_dim]
        return self.head(x)
