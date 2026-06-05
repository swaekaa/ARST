"""
Weights & Biases utilities for ARST.

Provides thin wrappers around the W&B Python SDK that:
    1. Handle graceful fallback when W&B is unavailable / disabled.
    2. Standardise run initialisation with project metadata.
    3. Offer a :class:`WandbLogger` class for in-loop metric logging.

All training code calls :class:`WandbLogger` instead of ``wandb.*``
directly so that W&B can be disabled without touching training scripts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed — W&B logging disabled. Install with: pip install wandb")


def init_wandb(
    project: str = "arst-behavior-recognition",
    entity: str | None = None,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
    enabled: bool = True,
) -> Any | None:
    """
    Initialise a W&B run.

    Args:
        project:  W&B project name.
        entity:   W&B entity (team/username). ``None`` uses default.
        name:     Run display name. ``None`` auto-generates one.
        config:   Hyperparameter dict logged to W&B Config.
        tags:     List of string tags.
        mode:     ``"online"`` | ``"offline"`` | ``"disabled"``.
        enabled:  If ``False``, immediately returns ``None`` (no-op).

    Returns:
        W&B run object, or ``None`` if W&B is unavailable/disabled.
    """
    if not enabled or not _WANDB_AVAILABLE:
        return None

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            tags=tags or [],
            mode=mode,
            reinit=True,
        )
        logger.info("W&B run initialised: %s (project=%s)", name, project)
        return run
    except Exception as exc:
        logger.warning("W&B initialisation failed: %s — continuing without W&B.", exc)
        return None


def finish_wandb(run: Any | None = None) -> None:
    """
    Finish the active (or specified) W&B run.

    Safe to call even if W&B is disabled.
    """
    if not _WANDB_AVAILABLE:
        return
    try:
        if run is not None:
            run.finish()
        else:
            wandb.finish()
    except Exception as exc:
        logger.warning("W&B finish failed: %s", exc)


class WandbLogger:
    """
    Thin logging wrapper that silently no-ops when W&B is unavailable.

    Args:
        run:      W&B run object returned by :func:`init_wandb`. ``None`` → all calls are no-ops.
        prefix:   Optional prefix prepended to every logged key.
    """

    def __init__(self, run: Any | None = None, prefix: str = "") -> None:
        self.run = run
        self.prefix = prefix
        self._step: int = 0

    @property
    def enabled(self) -> bool:
        """True if an active W&B run is attached."""
        return self.run is not None and _WANDB_AVAILABLE

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log a dict of metrics to W&B.

        Args:
            metrics: Metric name → value dict.
            step:    Global step counter. If ``None``, uses internal counter.
        """
        if not self.enabled:
            return
        _step = step if step is not None else self._step
        prefixed = {f"{self.prefix}{k}" if self.prefix else k: v for k, v in metrics.items()}
        try:
            self.run.log(prefixed, step=_step)
        except Exception as exc:
            logger.warning("W&B log failed: %s", exc)
        self._step = _step + 1

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """
        Write metrics to the W&B run summary (shown in the run table).

        Args:
            metrics: Final metrics dict (e.g. test results).
        """
        if not self.enabled:
            return
        try:
            for k, v in metrics.items():
                self.run.summary[f"{self.prefix}{k}" if self.prefix else k] = v
        except Exception as exc:
            logger.warning("W&B summary failed: %s", exc)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """
        Upload a file or directory as a W&B Artifact.

        Args:
            path:          Local path to the file/directory.
            name:          Artifact name.
            artifact_type: W&B artifact type tag.
        """
        if not self.enabled:
            return
        try:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)
            logger.info("W&B: uploaded artifact '%s' from %s", name, path)
        except Exception as exc:
            logger.warning("W&B artifact upload failed: %s", exc)

    def watch(self, model: Any, log_freq: int = 100) -> None:
        """
        Attach W&B gradient/parameter tracking to a model.

        Args:
            model:    PyTorch model.
            log_freq: Log histograms every N steps.
        """
        if not self.enabled:
            return
        try:
            self.run.watch(model, log="gradients", log_freq=log_freq)
        except Exception as exc:
            logger.warning("W&B watch failed: %s", exc)
