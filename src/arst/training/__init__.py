"""
Training package for ARST.

Public API:
    - :class:`~arst.training.trainer.Trainer`
    - :class:`~arst.training.metrics.MetricsCalculator`
    - :class:`~arst.training.losses.FocalLoss`
    - :func:`~arst.training.losses.build_loss`
    - :class:`~arst.training.callbacks.EarlyStopping`
    - :class:`~arst.training.callbacks.ModelCheckpoint`
    - :class:`~arst.training.callbacks.MetricTracker`
    - :func:`~arst.training.checkpointing.save_best`
    - :func:`~arst.training.checkpointing.save_last`
    - :func:`~arst.training.checkpointing.load_checkpoint`
    - :class:`~arst.training.experiment.ExperimentContext`
    - :func:`~arst.training.experiment.seed_everything`
    - :func:`~arst.training.experiment.get_device`
    - :func:`~arst.training.trainer.build_optimizer`
    - :func:`~arst.training.trainer.build_scheduler`
"""

from arst.training.callbacks import EarlyStopping, MetricTracker, ModelCheckpoint
from arst.training.checkpointing import load_checkpoint, save_best, save_last
from arst.training.experiment import ExperimentContext, get_device, seed_everything
from arst.training.losses import FocalLoss, build_loss
from arst.training.metrics import MetricsCalculator
from arst.training.trainer import Trainer, build_optimizer, build_scheduler

__all__ = [
    "Trainer",
    "MetricsCalculator",
    "FocalLoss",
    "build_loss",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricTracker",
    "save_best",
    "save_last",
    "load_checkpoint",
    "ExperimentContext",
    "seed_everything",
    "get_device",
    "build_optimizer",
    "build_scheduler",
]
