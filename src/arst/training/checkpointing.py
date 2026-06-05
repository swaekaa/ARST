"""
Checkpointing utilities for ARST Phase 2.

Provides:
    - :func:`save_best`: Save a checkpoint when it beats the current best metric.
    - :func:`save_last`: Unconditionally save the latest epoch state.
    - :func:`load_checkpoint`: Restore model (and optionally optimizer) from a file.

These are thin helpers that wrap raw :mod:`torch` save/load so that other
modules don't hardcode checkpoint logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_best(
    model: nn.Module,
    metric_value: float,
    best_value: float,
    mode: str,
    checkpoint_dir: Path,
    epoch: int,
    model_name: str = "model",
    extra: dict[str, Any] | None = None,
) -> tuple[float, Path | None]:
    """
    Save a checkpoint if ``metric_value`` beats ``best_value``.

    Args:
        model:           The PyTorch model to checkpoint.
        metric_value:    Current epoch metric value.
        best_value:      Best metric value seen so far.
        mode:            ``"max"`` (higher is better) or ``"min"``.
        checkpoint_dir:  Directory to write checkpoint files.
        epoch:           Current epoch index (0-based).
        model_name:      Name prefix for the checkpoint filename.
        extra:           Optional dict of additional data to store in the
                         checkpoint (e.g. optimizer state, config, metrics).

    Returns:
        Tuple of ``(new_best_value, saved_path)`` where ``saved_path``
        is ``None`` if no improvement was detected.
    """
    is_better = (metric_value > best_value) if mode == "max" else (metric_value < best_value)
    if not is_better:
        return best_value, None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{model_name}_best.pt"

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": metric_value,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    logger.info(
        "save_best: new best %.4f → %.4f at epoch %d — saved to %s",
        best_value,
        metric_value,
        epoch,
        path,
    )
    return metric_value, path


def save_last(
    model: nn.Module,
    checkpoint_dir: Path,
    epoch: int,
    model_name: str = "model",
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Unconditionally save the current model state as the "last" checkpoint.

    Args:
        model:           The PyTorch model to checkpoint.
        checkpoint_dir:  Directory to write the checkpoint.
        epoch:           Current epoch index (0-based).
        model_name:      Name prefix for the checkpoint filename.
        extra:           Optional dict of additional data (same as :func:`save_best`).

    Returns:
        Path of the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{model_name}_last.pt"

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    logger.debug("save_last: epoch %d → %s", epoch, path)
    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint into ``model`` (and optionally ``optimizer``).

    Args:
        model:            Model to load weights into.
        checkpoint_path:  Path to the ``.pt`` file.
        optimizer:        If provided, restore optimizer state too.
        device:           Target device. Defaults to the model's current device.
        strict:           Passed to :meth:`~torch.nn.Module.load_state_dict`.

    Returns:
        The full checkpoint payload dict (includes ``epoch``, ``metrics``, etc.).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or next(model.parameters()).device
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    logger.info("load_checkpoint: loaded '%s' (epoch=%s)", path.name, payload.get("epoch", "?"))

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        logger.info("load_checkpoint: optimizer state restored")

    return payload
