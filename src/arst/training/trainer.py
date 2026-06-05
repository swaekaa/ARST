"""
Core Trainer for ARST Phase 2.

Implements a clean, self-contained training loop that supports:
    - Mixed-precision training (AMP) for RTX 3060 4 GB
    - Gradient accumulation (effective batch = batch_size × accumulation_steps)
    - Gradient clipping
    - Callback system (EarlyStopping, ModelCheckpoint, MetricTracker)
    - Weights & Biases logging (optional, gracefully disabled if W&B unavailable)
    - Progress bars via tqdm
    - train_epoch() / validate_epoch() / test_epoch() public API

Design constraints from Phase 1:
    - RTX 3060 4 GB VRAM → AMP mandatory for larger models
    - Class imbalance 3.79× → Focal Loss + MetricsCalculator(Macro F1)
    - CSV loaders are already subject-stratified splits
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler  # type: ignore[attr-defined]
from torch.utils.data import DataLoader
from tqdm import tqdm

from arst.training.callbacks import Callback, EarlyStopping, MetricTracker
from arst.training.checkpointing import save_best, save_last
from arst.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Self-contained training loop for all ARST baseline and future models.

    Args:
        model:              PyTorch model (any :class:`~torch.nn.Module`).
        optimizer:          Optimiser instance.
        loss_fn:            Loss module (e.g. :class:`~arst.training.losses.FocalLoss`).
        device:             Target device.
        num_classes:        Number of behavior classes (4).
        max_epochs:         Maximum training epochs.
        accumulation_steps: Gradient accumulation steps (effective batch multiplier).
        mixed_precision:    Enable AMP; strongly recommended for RTX 3060.
        grad_clip_norm:     Max gradient norm (``None`` to disable).
        callbacks:          List of :class:`~arst.training.callbacks.Callback` instances.
        scheduler:          Optional LR scheduler (called once per epoch after validation).
        wandb_run:          Optional W&B run object for metric logging.
        log_every_n_steps:  Log step-level metrics every N optimizer steps.
        class_names:        Class name strings for MetricsCalculator.
        checkpoint_dir:     Directory for checkpoints (used by :func:`save_last` / :func:`save_best`).
        monitor_metric:     Metric key used to decide best checkpoint (e.g. ``"val/f1_macro"``).
        monitor_mode:       ``"max"`` or ``"min"``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        num_classes: int = 4,
        max_epochs: int = 100,
        accumulation_steps: int = 1,
        mixed_precision: bool = True,
        grad_clip_norm: float | None = 1.0,
        callbacks: list[Callback] | None = None,
        scheduler: Any | None = None,
        wandb_run: Any | None = None,
        log_every_n_steps: int = 10,
        class_names: list[str] | None = None,
        checkpoint_dir: str | Path = "experiments/default/checkpoints",
        monitor_metric: str = "val/f1_macro",
        monitor_mode: str = "max",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.accumulation_steps = max(1, accumulation_steps)
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.callbacks: list[Callback] = callbacks or []
        self.scheduler = scheduler
        self.wandb_run = wandb_run
        self.log_every_n_steps = log_every_n_steps
        self.class_names = class_names
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        self.scaler = GradScaler(enabled=self.mixed_precision)
        self._train_metrics = MetricsCalculator(num_classes=num_classes, class_names=class_names)
        self._val_metrics = MetricsCalculator(num_classes=num_classes, class_names=class_names)
        self._best_metric: float = float("-inf") if monitor_mode == "max" else float("inf")
        self._global_step: int = 0

        # Locate EarlyStopping callback if present
        self._early_stopping: EarlyStopping | None = next(
            (cb for cb in self.callbacks if isinstance(cb, EarlyStopping)), None
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """
        Train for up to ``max_epochs`` epochs, running validation after each.

        Args:
            train_loader: Training :class:`~torch.utils.data.DataLoader`.
            val_loader:   Validation :class:`~torch.utils.data.DataLoader`.

        Returns:
            History dict: ``{metric_name: [epoch0_value, epoch1_value, …]}``.
        """
        tracker: MetricTracker | None = next(
            (cb for cb in self.callbacks if isinstance(cb, MetricTracker)), None
        )

        logger.info(
            "Training start: model=%s  device=%s  epochs=%d  amp=%s",
            self.model.__class__.__name__,
            self.device,
            self.max_epochs,
            self.mixed_precision,
        )

        for epoch in range(self.max_epochs):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch)
            epoch_time = time.perf_counter() - t0

            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics, "epoch_time_s": epoch_time}

            # LR scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                all_metrics["learning_rate"] = current_lr
                if self.wandb_run:
                    self.wandb_run.log({"learning_rate": current_lr}, step=self._global_step)

            # Callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, all_metrics, self.model)

            # Checkpointing
            val_monitor_value = all_metrics.get(self.monitor_metric, None)
            if val_monitor_value is not None:
                self._best_metric, _ = save_best(
                    model=self.model,
                    metric_value=val_monitor_value,
                    best_value=self._best_metric,
                    mode=self.monitor_mode,
                    checkpoint_dir=self.checkpoint_dir,
                    epoch=epoch,
                    extra={
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": all_metrics,
                    },
                )
            save_last(
                model=self.model,
                checkpoint_dir=self.checkpoint_dir,
                epoch=epoch,
                extra={"optimizer_state_dict": self.optimizer.state_dict()},
            )

            # Log epoch summary
            log_line = (
                f"Epoch {epoch:03d}/{self.max_epochs - 1}  "
                f"train_loss={train_metrics.get('train/loss', 0):.4f}  "
                f"val_loss={val_metrics.get('val/loss', 0):.4f}  "
                f"val_f1={val_metrics.get('val/f1_macro', 0):.4f}  "
                f"[{epoch_time:.1f}s]"
            )
            logger.info(log_line)

            # W&B epoch-level log
            if self.wandb_run:
                self.wandb_run.log(all_metrics, step=self._global_step)

            # Early stopping check
            if self._early_stopping is not None and self._early_stopping.should_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        for cb in self.callbacks:
            cb.on_train_end(self.model)

        return tracker.history if tracker else {}

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        Execute one training epoch.

        Args:
            loader: Training DataLoader.
            epoch:  Current epoch index.

        Returns:
            Dict with ``"train/loss"`` and ``"train/f1_macro"``.
        """
        self.model.train()
        self._train_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            # Move batch to device
            imu = batch["imu"].to(self.device, non_blocking=True)
            thermo = batch["thermo"].to(self.device, non_blocking=True)
            tof = batch["tof"].to(self.device, non_blocking=True)
            tof_mask = batch.get("tof_mask")
            if tof_mask is not None:
                tof_mask = tof_mask.to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                logits = self.model(imu=imu, thermo=thermo, tof=tof, tof_mask=tof_mask)
                loss = self.loss_fn(logits, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self._global_step += 1

                if self.wandb_run and self._global_step % self.log_every_n_steps == 0:
                    self.wandb_run.log(
                        {"train/loss_step": loss.item() * self.accumulation_steps},
                        step=self._global_step,
                    )

            self._train_metrics.update(logits, labels)
            batch_loss = loss.item() * self.accumulation_steps
            total_loss += batch_loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        # Handle remainder gradient step
        if len(loader) % self.accumulation_steps != 0:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        computed = self._train_metrics.compute_and_reset()
        return {
            "train/loss": total_loss / max(n_batches, 1),
            "train/f1_macro": computed["f1_macro"],
            "train/accuracy": computed["accuracy"],
        }

    def validate_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        Execute one validation pass (no gradient computation).

        Args:
            loader: Validation DataLoader.
            epoch:  Current epoch index.

        Returns:
            Dict with ``"val/loss"``, ``"val/f1_macro"``, ``"val/f1_weighted"``,
            ``"val/accuracy"``.
        """
        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"  Val  E{epoch:03d}", leave=False, dynamic_ncols=True)
            for batch in pbar:
                imu = batch["imu"].to(self.device, non_blocking=True)
                thermo = batch["thermo"].to(self.device, non_blocking=True)
                tof = batch["tof"].to(self.device, non_blocking=True)
                tof_mask = batch.get("tof_mask")
                if tof_mask is not None:
                    tof_mask = tof_mask.to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    logits = self.model(imu=imu, thermo=thermo, tof=tof, tof_mask=tof_mask)
                    loss = self.loss_fn(logits, labels)

                self._val_metrics.update(logits, labels)
                total_loss += loss.item()
                n_batches += 1

        computed = self._val_metrics.compute_and_reset()
        return {
            "val/loss": total_loss / max(n_batches, 1),
            "val/f1_macro": computed["f1_macro"],
            "val/f1_weighted": computed["f1_weighted"],
            "val/accuracy": computed["accuracy"],
        }

    def test_epoch(self, loader: DataLoader) -> dict[str, Any]:
        """
        Run inference on test set; returns full metrics dict including
        confusion matrix.

        Args:
            loader: Test DataLoader.

        Returns:
            Full metrics dict from :class:`~arst.training.metrics.MetricsCalculator`.
        """
        self.model.eval()
        test_metrics = MetricsCalculator(num_classes=self.num_classes, class_names=self.class_names)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Test", dynamic_ncols=True):
                imu = batch["imu"].to(self.device, non_blocking=True)
                thermo = batch["thermo"].to(self.device, non_blocking=True)
                tof = batch["tof"].to(self.device, non_blocking=True)
                tof_mask = batch.get("tof_mask")
                if tof_mask is not None:
                    tof_mask = tof_mask.to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    logits = self.model(imu=imu, thermo=thermo, tof=tof, tof_mask=tof_mask)
                    loss = self.loss_fn(logits, labels)

                test_metrics.update(logits, labels)
                total_loss += loss.item()
                n_batches += 1

        result = test_metrics.compute()
        result["test/loss"] = total_loss / max(n_batches, 1)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────────────────────────────────────


def build_optimizer(
    model: nn.Module,
    optimizer_cfg: dict | Any,
) -> torch.optim.Optimizer:
    """
    Build an optimiser from a config dict or OmegaConf object.

    Supported types: ``"adamw"`` (default), ``"adam"``, ``"sgd"``.

    Args:
        model:          Model whose parameters to optimise.
        optimizer_cfg:  Config with at least a ``type`` key.

    Returns:
        Configured :class:`~torch.optim.Optimizer`.
    """
    cfg: dict = (
        optimizer_cfg
        if isinstance(optimizer_cfg, dict)
        else {k: v for k, v in vars(optimizer_cfg).items() if not k.startswith("_")}
    )

    opt_type: str = str(cfg.get("type", "adamw")).lower()
    lr: float = float(cfg.get("lr", 1e-4))
    weight_decay: float = float(cfg.get("weight_decay", 1e-2))

    if opt_type == "adamw":
        betas = tuple(cfg.get("betas", [0.9, 0.999]))
        eps = float(cfg.get("eps", 1e-8))
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
    elif opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict | Any,
    max_epochs: int,
) -> Any | None:
    """
    Build a learning-rate scheduler.

    Supported types: ``"cosine_with_warmup"``, ``"cosine"``, ``"step"``, ``"none"``.
    """
    cfg: dict = (
        scheduler_cfg
        if isinstance(scheduler_cfg, dict)
        else {k: v for k, v in vars(scheduler_cfg).items() if not k.startswith("_")}
    )
    sched_type: str = str(cfg.get("type", "cosine_with_warmup")).lower()

    if sched_type in ("none", ""):
        return None
    elif sched_type == "cosine":
        eta_min = float(cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=eta_min
        )
    elif sched_type == "cosine_with_warmup":
        warmup_epochs = int(cfg.get("warmup_epochs", 5))
        eta_min = float(cfg.get("min_lr", 1e-6))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            import math

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min / optimizer.param_groups[0]["initial_lr"] + cosine * (
                1 - eta_min / optimizer.param_groups[0]["initial_lr"]
            )

        # Store initial_lr in param groups for lambda reference
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_type == "step":
        step_size = int(cfg.get("step_size", 30))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type!r}")
