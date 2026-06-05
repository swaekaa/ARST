"""
Training callbacks for ARST Phase 2.

Provides:
    - :class:`EarlyStopping`: Stop training when a metric stagnates.
    - :class:`ModelCheckpoint`: Save top-k checkpoints based on a metric.
    - :class:`MetricTracker`: Record metric history for plotting/reporting.

All callbacks share a minimal interface so the Trainer can call them uniformly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────


class Callback:
    """Abstract base class for training callbacks."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        """Called at the end of every epoch."""

    def on_train_end(self, model: nn.Module) -> None:
        """Called once when training finishes or is stopped."""


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric does not improve for ``patience``
    consecutive epochs.

    Args:
        monitor: Metric name to watch (must be a key in ``metrics`` dict).
        patience: Number of epochs to wait after last improvement.
        mode: ``"max"`` (higher is better) or ``"min"`` (lower is better).
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "val/f1_macro",
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._counter: int = 0
        self._best_value: float = float("-inf") if mode == "max" else float("inf")
        self.stopped: bool = False
        self.stopped_epoch: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self._best_value + self.min_delta
        return current < self._best_value - self.min_delta

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            logger.warning("EarlyStopping: metric '%s' not found in metrics dict.", self.monitor)
            return

        if self._is_improvement(value):
            self._best_value = value
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: no improvement in '%s' for %d / %d epochs (best=%.4f, current=%.4f)",
                self.monitor,
                self._counter,
                self.patience,
                self._best_value,
                value,
            )
            if self._counter >= self.patience:
                self.stopped = True
                self.stopped_epoch = epoch
                logger.info(
                    "EarlyStopping: stopping training at epoch %d (best %s=%.4f).",
                    epoch,
                    self.monitor,
                    self._best_value,
                )

    @property
    def should_stop(self) -> bool:
        """Returns True if training should be halted."""
        return self.stopped


# ──────────────────────────────────────────────────────────────────────────────
# Model Checkpoint
# ──────────────────────────────────────────────────────────────────────────────


class ModelCheckpoint(Callback):
    """
    Save model checkpoints when a monitored metric improves.

    Keeps the top-k checkpoints on disk and always saves the ``last.pt``
    checkpoint regardless of performance.

    Args:
        dirpath: Directory in which checkpoints are saved.
        monitor: Metric key to watch.
        mode: ``"max"`` or ``"min"``.
        save_top_k: Maximum number of best checkpoints to retain.
        filename_prefix: Filename prefix for checkpoint files.
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "val/f1_macro",
        mode: str = "max",
        save_top_k: int = 3,
        filename_prefix: str = "ckpt",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_prefix = filename_prefix
        # List of (value, path) tuples, maintained sorted
        self._saved: list[tuple[float, Path]] = []
        self.best_model_path: Path | None = None

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = (
            self.dirpath
            / f"{self.filename_prefix}_epoch{epoch:03d}_{self.monitor.replace('/', '_')}={value:.4f}.pt"
        )

        # Always save current as "last"
        last_path = self.dirpath / "last.pt"
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, last_path
        )

        # Determine whether this is a top-k checkpoint
        should_save = len(self._saved) < self.save_top_k
        if not should_save:
            # Compare against the worst saved checkpoint
            worst_val, worst_path = (
                min(self._saved, key=lambda x: x[0])
                if self.mode == "max"
                else max(self._saved, key=lambda x: x[0])
            )
            should_save = self._is_better(value, worst_val)
            if should_save:
                worst_path.unlink(missing_ok=True)
                self._saved.remove((worst_val, worst_path))

        if should_save:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics},
                ckpt_path,
            )
            self._saved.append((value, ckpt_path))
            logger.info("ModelCheckpoint: saved %s (metric=%.4f)", ckpt_path.name, value)

            # Track best
            if self.best_model_path is None or self._is_better(
                value,
                metrics.get(self.monitor, float("-inf") if self.mode == "max" else float("inf")),
            ):
                self.best_model_path = ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Metric Tracker
# ──────────────────────────────────────────────────────────────────────────────


class MetricTracker(Callback):
    """
    Records metric history for every epoch.

    After training, access :attr:`history` to retrieve a dict mapping
    metric name → list of per-epoch values.
    """

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> None:
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value))

    def get(self, key: str) -> list[float]:
        """Return the recorded history for a single metric key."""
        return self.history.get(key, [])

    def best(self, key: str, mode: str = "max") -> tuple[int, float]:
        """
        Return ``(epoch_idx, best_value)`` for a metric.

        Args:
            key:  Metric name.
            mode: ``"max"`` or ``"min"``.
        """
        values = self.history.get(key, [])
        if not values:
            raise KeyError(f"Metric '{key}' not found in history.")
        fn = max if mode == "max" else min
        best_val = fn(values)
        return values.index(best_val), best_val

    def summary(self) -> dict[str, Any]:
        """Return a flat dict of the best value for each tracked metric."""
        result: dict[str, Any] = {}
        for key, values in self.history.items():
            result[f"{key}_best"] = max(values)
            result[f"{key}_last"] = values[-1]
        return result
