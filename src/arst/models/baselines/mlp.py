"""
MLP Baseline for ARST Phase 2.

Architecture:
    Input: [B, T, C_imu] + [B, T, C_thm] + [B, T, C_tof]
        → Temporal mean pooling per modality
        → Concatenate active modality features
        → BatchNorm → Linear → GELU → Dropout (× N layers)
        → Linear → logits [B, num_classes]

Design notes:
    - Correct Phase 1 dims: IMU=7, Thermal=5, ToF=320.
    - ``active_modalities`` controls which modalities are used.
      This is critical for Phase 7 ablation studies (unimodal comparisons).
    - Default: all three modalities concatenated (332 features).
    - ToF mask is accepted but ignored by this baseline (no reliability).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Phase 1 verified channel counts
_IMU_CH = 7
_THM_CH = 5
_TOF_CH = 320


class MLPBaseline(nn.Module):
    """
    MLP Baseline: temporal mean pooling + multi-layer perceptron.

    No temporal modelling — treats each window as a fixed feature vector.
    Serves as the lowest learned-model performance reference.

    Args:
        num_classes:       Number of behavior classes (4 from Phase 1 EDA).
        imu_channels:      IMU feature channels (7: acc_xyz + quaternion).
        thermal_channels:  Thermopile channels (5: linear array).
        tof_channels:      ToF channels (320: 5 sensors × 64 pixels).
        hidden_dims:       MLP hidden layer widths (list).
        dropout:           Dropout probability applied after each hidden layer.
        active_modalities: Which modalities to use.  Controls input dimension.
            Options: ``["imu"]``, ``["thermo"]``, ``["tof"]``,
            ``["imu", "thermo", "tof"]`` (default — early fusion).
    """

    def __init__(
        self,
        num_classes: int = 4,
        imu_channels: int = _IMU_CH,
        thermal_channels: int = _THM_CH,
        tof_channels: int = _TOF_CH,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        active_modalities: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.active_modalities: list[str] = (
            active_modalities if active_modalities is not None else ["imu", "thermo", "tof"]
        )
        self.imu_channels = imu_channels
        self.thermal_channels = thermal_channels
        self.tof_channels = tof_channels

        # Compute input feature size based on active modalities
        in_features = 0
        if "imu" in self.active_modalities:
            in_features += imu_channels
        if "thermo" in self.active_modalities:
            in_features += thermal_channels
        if "tof" in self.active_modalities:
            in_features += tof_channels

        if in_features == 0:
            raise ValueError("At least one modality must be active.")

        _hidden_dims: list[int] = hidden_dims if hidden_dims is not None else [512, 256, 128]

        # Build MLP
        layers: list[nn.Module] = []
        prev_dim = in_features
        for hdim in _hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

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
            imu:      [B, T, 7]    IMU sequence (acc + quaternion).
            thermo:   [B, T, 5]    Thermopile sequence (may be None if inactive).
            tof:      [B, T, 320]  ToF sequence (may be None if inactive).
            tof_mask: [B, T, 320]  ToF validity mask — accepted but ignored
                                   (no reliability module in Phase 2).

        Returns:
            logits: [B, num_classes]
        """
        parts: list[torch.Tensor] = []

        if "imu" in self.active_modalities:
            assert imu is not None, "IMU tensor required but not provided."
            parts.append(imu.mean(dim=1))  # [B, 7]

        if "thermo" in self.active_modalities:
            assert thermo is not None, "Thermal tensor required but not provided."
            parts.append(thermo.mean(dim=1))  # [B, 5]

        if "tof" in self.active_modalities:
            assert tof is not None, "ToF tensor required but not provided."
            parts.append(tof.mean(dim=1))  # [B, 320]

        x = torch.cat(parts, dim=-1)  # [B, in_features]
        return self.mlp(x)  # [B, num_classes]
