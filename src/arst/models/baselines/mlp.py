"""
Baseline MLP model for behavior recognition.

Flattens all modality features into a single vector, applies statistical
aggregation over the time axis, then runs a multi-layer perceptron.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    MLP Baseline: temporal mean pooling + multi-layer perceptron.

    No temporal modeling — treats each sequence as a fixed feature vector.
    Serves as a lower-bound performance reference.

    Args:
        imu_channels: IMU feature channels (default 6).
        thermal_channels: Thermopile channels (default 64).
        tof_channels: ToF channels (default 64).
        hidden_dims: MLP hidden layer sizes.
        num_classes: Number of behavior classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        imu_channels: int = 6,
        thermal_channels: int = 64,
        tof_channels: int = 64,
        hidden_dims: list[int] = [512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_features = imu_channels + thermal_channels + tof_channels

        layers = []
        prev_dim = in_features
        for hdim in hidden_dims:
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
        # Temporal mean pooling
        imu_feat = imu.mean(dim=1)  # [B, 6]
        therm_feat = thermo.mean(dim=1)  # [B, 64]
        tof_feat = tof.mean(dim=1)  # [B, 64]

        # Concatenate all modalities
        x = torch.cat([imu_feat, therm_feat, tof_feat], dim=-1)  # [B, 134]

        return self.mlp(x)  # [B, num_classes]
