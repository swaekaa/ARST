"""
ARST Phase 2 — Training Entry Point.

Hydra-driven training script supporting all baseline models.

Usage::

    # Train MLP baseline (all modalities)
    python train.py model=mlp

    # Train CNN baseline
    python train.py model=cnn

    # Train BiLSTM baseline
    python train.py model=lstm

    # Train Transformer baseline
    python train.py model=transformer

    # Unimodal ablation (IMU only)
    python train.py model=mlp model.active_modalities=[imu]

    # Override training params
    python train.py model=transformer training.epochs=50 training.batch_size=16

    # Enable W&B
    python train.py model=cnn wandb.enabled=true

    # Quick debug run (50k rows only)
    python train.py model=mlp data.max_rows=50000 training.epochs=3

    # Sanity baseline
    python train.py model=majority training.epochs=1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from arst.data.dataloader import build_csv_loaders
from arst.models.registry import get_model
from arst.training.callbacks import EarlyStopping, MetricTracker, ModelCheckpoint
from arst.training.experiment import ExperimentContext, make_run_id
from arst.training.losses import build_loss
from arst.training.metrics import CLASS_NAMES
from arst.training.trainer import Trainer, build_optimizer, build_scheduler
from arst.utils.wandb_utils import WandbLogger, finish_wandb, init_wandb

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point — driven by Hydra configuration.

    Args:
        cfg: Hydra-composed DictConfig.
    """
    # ── Print resolved config ──────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("ARST Phase 2 — Training")
    logger.info("=" * 70)
    logger.info("\n%s", OmegaConf.to_yaml(cfg))

    # ── Experiment context (seed, device, dirs) ────────────────────────────
    model_name: str = cfg.model.name
    seed: int = int(cfg.get("seed", 42))
    run_id = make_run_id(model_name, seed)

    experiment_dir = Path(cfg.get("experiment_dir", "experiments"))
    ctx = ExperimentContext(
        run_id=run_id,
        base_dir=experiment_dir,
        seed=seed,
        deterministic=bool(cfg.get("deterministic", False)),
    )

    # ── W&B setup ─────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled: bool = bool(wandb_cfg.get("enabled", False))
    wb_run = init_wandb(
        project=str(wandb_cfg.get("project", "arst-behavior-recognition")),
        entity=wandb_cfg.get("entity", None),
        name=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(wandb_cfg.get("tags", [])) + [model_name, "phase2"],
        enabled=wandb_enabled,
    )
    wb_logger = WandbLogger(run=wb_run)

    try:
        # ── DataLoaders ───────────────────────────────────────────────────
        data_cfg = cfg.data
        csv_path = Path(data_cfg.csv_path)

        if not csv_path.exists():
            logger.error(
                "Training CSV not found: %s\n"
                "Run: python scripts/download_data.py  to fetch the dataset.",
                csv_path,
            )
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        active_modalities: list[str] = list(
            data_cfg.get("active_modalities", ["imu", "thermo", "tof"])
        )
        logger.info("Active modalities: %s", active_modalities)

        train_loader, val_loader, test_loader, data_info = build_csv_loaders(
            csv_path=csv_path,
            window_size=int(data_cfg.window_size),
            batch_size=int(data_cfg.batch_size),
            val_fraction=float(data_cfg.get("val_fraction", 0.15)),
            test_fraction=float(data_cfg.get("test_fraction", 0.15)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=seed,
            max_rows=data_cfg.get("max_rows", None),
        )
        n_classes: int = data_info["n_classes"]
        class_weights: torch.Tensor = data_info["class_weights"].to(ctx.device)

        logger.info(
            "Loaders ready: train=%d  val=%d  test=%d  classes=%d",
            data_info["n_train"],
            data_info["n_val"],
            data_info["n_test"],
            n_classes,
        )

        # ── Model ─────────────────────────────────────────────────────────
        model_kwargs: dict = OmegaConf.to_container(cfg.model, resolve=True)  # type: ignore[assignment]
        model_kwargs.pop("name")  # already used as registry key
        model_kwargs["num_classes"] = n_classes
        model_kwargs["active_modalities"] = active_modalities

        model = get_model(model_name, **model_kwargs)
        model = model.to(ctx.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model: %s  |  Trainable params: %s", model.__class__.__name__, f"{n_params:,}")

        if wb_logger.enabled:
            wb_logger.watch(model, log_freq=100)

        # ── Handle non-trainable baselines ────────────────────────────────
        if model_name in ("majority", "random"):
            _run_non_trainable_baseline(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=ctx.device,
                n_classes=n_classes,
                output_dir=Path(cfg.get("output_dir", "outputs/evaluation")),
                wb_logger=wb_logger,
            )
            return

        # ── Loss ──────────────────────────────────────────────────────────
        training_cfg = cfg.training
        loss_fn = build_loss(training_cfg.loss, class_weights=class_weights)

        # ── Optimizer & Scheduler ─────────────────────────────────────────
        optimizer = build_optimizer(model, training_cfg.optimizer)
        max_epochs: int = int(training_cfg.epochs)
        scheduler = build_scheduler(optimizer, training_cfg.scheduler, max_epochs)

        # ── Callbacks ─────────────────────────────────────────────────────
        es_cfg = training_cfg.early_stopping
        early_stopping = (
            EarlyStopping(
                monitor=str(es_cfg.monitor),
                patience=int(es_cfg.patience),
                mode=str(es_cfg.mode),
                min_delta=float(es_cfg.min_delta),
            )
            if bool(es_cfg.enabled)
            else None
        )

        ckpt_cfg = training_cfg.checkpointing
        model_checkpoint = (
            ModelCheckpoint(
                dirpath=ctx.checkpoint_dir,
                monitor=str(ckpt_cfg.monitor),
                mode=str(ckpt_cfg.mode),
                save_top_k=int(ckpt_cfg.save_top_k),
                filename_prefix=model_name,
            )
            if bool(ckpt_cfg.enabled)
            else None
        )

        metric_tracker = MetricTracker()

        callbacks = [
            cb for cb in [early_stopping, model_checkpoint, metric_tracker] if cb is not None
        ]

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=ctx.device,
            num_classes=n_classes,
            max_epochs=max_epochs,
            accumulation_steps=int(training_cfg.accumulation_steps),
            mixed_precision=bool(training_cfg.mixed_precision),
            grad_clip_norm=float(training_cfg.gradient_clipping.max_norm),
            callbacks=callbacks,
            scheduler=scheduler,
            wandb_run=wb_run,
            log_every_n_steps=int(training_cfg.wandb.log_freq),
            class_names=CLASS_NAMES,
            checkpoint_dir=ctx.checkpoint_dir,
            monitor_metric=str(ckpt_cfg.monitor),
            monitor_mode=str(ckpt_cfg.mode),
        )

        # ── Train ─────────────────────────────────────────────────────────
        logger.info("Starting training: %d epochs  |  run_id=%s", max_epochs, run_id)
        history = trainer.fit(train_loader, val_loader)

        # ── Test evaluation ───────────────────────────────────────────────
        logger.info("Running test evaluation...")
        test_metrics = trainer.test_epoch(test_loader)
        logger.info(
            "Test results: accuracy=%.4f  f1_macro=%.4f  f1_weighted=%.4f",
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
            test_metrics["f1_weighted"],
        )

        # Save test results
        from arst.evaluation.evaluate import _save_evaluation_outputs

        eval_out_dir = Path(cfg.get("output_dir", "outputs/evaluation")) / run_id
        _save_evaluation_outputs(test_metrics, eval_out_dir, "test")

        if wb_logger.enabled:
            wb_logger.log_summary(
                {f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, int | float)}
            )

        if model_checkpoint and model_checkpoint.best_model_path:
            logger.info("Best checkpoint: %s", model_checkpoint.best_model_path)

        if metric_tracker:
            _, best_val_f1 = metric_tracker.best("val/f1_macro", mode="max")
            logger.info("Best val F1 (macro): %.4f", best_val_f1)

    finally:
        finish_wandb(wb_run)


def _run_non_trainable_baseline(
    model: torch.nn.Module,
    model_name: str,
    train_loader: object,
    val_loader: object,
    test_loader: object,
    device: torch.device,
    n_classes: int,
    output_dir: Path,
    wb_logger: WandbLogger,
) -> None:
    """Run evaluation-only pipeline for Majority/Random baselines."""
    from arst.evaluation.evaluate import evaluate_model

    logger.info("Running non-trainable baseline: %s", model_name)

    # Collect all training labels to fit the baseline distribution
    all_labels: list[int] = []
    for batch in train_loader:  # type: ignore[union-attr]
        all_labels.extend(batch["label"].tolist())

    if hasattr(model, "fit"):
        model.fit(all_labels)
        logger.info("Baseline fitted to %d training labels.", len(all_labels))

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,  # type: ignore[arg-type]
        device=device,
        num_classes=n_classes,
        class_names=CLASS_NAMES,
        output_dir=output_dir / model_name,
        split_name="test",
    )
    logger.info(
        "[%s] test accuracy=%.4f  test f1_macro=%.4f",
        model_name,
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
    )
    if wb_logger.enabled:
        wb_logger.log_summary(
            {f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, int | float)}
        )


if __name__ == "__main__":
    main()
