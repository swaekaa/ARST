"""
Baselines package for ARST Phase 2.

All six baseline models for Phase 2 benchmarking:
    - :class:`~arst.models.baselines.majority.MajorityBaseline`
    - :class:`~arst.models.baselines.majority.RandomBaseline`
    - :class:`~arst.models.baselines.mlp.MLPBaseline`
    - :class:`~arst.models.baselines.cnn.CNNBaseline`
    - :class:`~arst.models.baselines.lstm.LSTMBaseline`
    - :class:`~arst.models.baselines.transformer.TransformerBaseline`
"""

from arst.models.baselines.cnn import CNNBaseline
from arst.models.baselines.lstm import LSTMBaseline
from arst.models.baselines.majority import MajorityBaseline, RandomBaseline
from arst.models.baselines.mlp import MLPBaseline
from arst.models.baselines.transformer import TransformerBaseline

__all__ = [
    "MajorityBaseline",
    "RandomBaseline",
    "MLPBaseline",
    "CNNBaseline",
    "LSTMBaseline",
    "TransformerBaseline",
]
