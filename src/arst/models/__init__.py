"""
Models package for ARST.

Includes:
  - Modality-specific encoders (IMU, Thermal, ToF)
  - Reliability module (ARM)
  - Fusion modules (Concat, Mean, Adaptive)
  - Classification heads
  - Full ARST model
  - Baseline models
"""

from arst.models.arst import ARSTModel
from arst.models.baselines.cnn import CNNBaseline
from arst.models.baselines.lstm import LSTMBaseline
from arst.models.baselines.mlp import MLPBaseline
from arst.models.baselines.transformer import TransformerBaseline

__all__ = [
    "ARSTModel",
    "MLPBaseline",
    "CNNBaseline",
    "LSTMBaseline",
    "TransformerBaseline",
]
