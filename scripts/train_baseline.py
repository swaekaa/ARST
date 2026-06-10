"""
ARST Phase 2.5 — Standalone Baseline Training Script.

This script trains a single baseline model WITHOUT Hydra, using hard-coded
configuration identical to the Hydra setup. Use this if Hydra is unavailable
or for simple one-line execution.

Usage (from repo root)::

    python scripts/train_baseline.py --model cnn
    python scripts/train_baseline.py --model lstm
    python scripts/train_baseline.py --model transformer
    python scripts/train_baseline.py --model mlp

This script produces IDENTICAL output to:
    python train.py model=<name>

Configuration mirrors configs/training/default.yaml exactly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_baseline")

# ─────────────────────────────────────────────────────────────────────────────
# Canonical training configuration (mirrors configs/training/default.yaml)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_CFG = {
    "epochs": 100,
    "seed": 42,
    "batch_size": 32,
    "accumulation_steps": 4,
    "mixed_precision": True,
    "optimizer": {
        "type": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    },
    "scheduler": {
        "type": "cosine_with_warmup",
        "warmup_epochs": 5,
        "min_lr": 1e-6,
    },
    "gradient_clipping": {"max_norm": 1.0},
    "loss": {
        "cls_type": "focal",
        "focal": {"gamma": 2.0, "reduction": "mean"},
        "use_class_weights": True,
    },
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "monitor": "val/f1_macro",
        "mode": "max",
        "min_delta": 0.001,
    },
    "checkpointing": {
        "enabled": True,
        "save_top_k": 3,
        "monitor": "val/f1_macro",
        "mode": "max",
    },
}

DATA_CFG = {
    "csv_path": "data/raw/train.csv",
    "window_size": 64,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "num_workers": 0,
    "active_modalities": ["imu", "thermo", "tof"],
}

OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation"
EXPERIMENT_DIR = REPO_ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ARST baseline model")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn", "lstm", "transformer", "mlp", "majority", "random"],
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply overrides
    cfg = dict(TRAIN_CFG)
    data_cfg = dict(DATA_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    model_name = args.model
    seed = args.seed

    logger.info("=" * 70)
    logger.info("ARST Phase 2.5 — Standalone Baseline Training")
    logger.info("=" * 70)
    logger.info("Model: %s", model_name)
    logger.info("Seed : %d", seed)
    logger.info("Epochs: %d", cfg["epochs"])

    import torch

    from arst.data.dataloader import build_csv_loaders
    from arst.models.registry import get_model
    from arst.training.callbacks import EarlyStopping, MetricTracker, ModelCheckpoint
    from arst.training.experiment import ExperimentContext, make_run_id
    from arst.training.losses import build_loss
    from arst.training.metrics import CLASS_NAMES
    from arst.training.trainer import Trainer, build_optimizer, build_scheduler

    # ── Experiment context ────────────────────────────────────────────────────
    run_id = make_run_id(model_name, seed)
    ctx = ExperimentContext(
        run_id=run_id,
        base_dir=str(EXPERIMENT_DIR),
        seed=seed,
        deterministic=False,
    )

    if args.no_cuda:
        ctx.device = torch.device("cpu")

    logger.info("Device : %s", ctx.device)
    logger.info("Run ID : %s", run_id)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    csv_path = REPO_ROOT / data_cfg["csv_path"]
    if not csv_path.exists():
        logger.error("Training CSV not found: %s", csv_path)
        logger.error("Run: python scripts/download_data.py  to fetch the dataset.")
        sys.exit(1)

    train_loader, val_loader, test_loader, data_info = build_csv_loaders(
        csv_path=csv_path,
        window_size=int(data_cfg["window_size"]),
        batch_size=int(cfg["batch_size"]),
        val_fraction=float(data_cfg["val_fraction"]),
        test_fraction=float(data_cfg["test_fraction"]),
        num_workers=int(data_cfg["num_workers"]),
        seed=seed,
        max_rows=None,
    )
    n_classes: int = data_info["n_classes"]
    class_weights: torch.Tensor = data_info["class_weights"].to(ctx.device)

    logger.info(
        "Loaders: train=%d  val=%d  test=%d  classes=%d",
        data_info["n_train"],
        data_info["n_val"],
        data_info["n_test"],
        n_classes,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(
        model_name,
        num_classes=n_classes,
        active_modalities=data_cfg["active_modalities"],
    ).to(ctx.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s  |  Params: %s", model.__class__.__name__, f"{n_params:,}")

    # ── Non-trainable baselines ───────────────────────────────────────────────
    if model_name in ("majority", "random"):
        from arst.evaluation.evaluate import evaluate_model

        all_labels: list[int] = []
        for batch in train_loader:
            all_labels.extend(batch["label"].tolist())

        if hasattr(model, "fit"):
            model.fit(all_labels)

        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=ctx.device,
            num_classes=n_classes,
            class_names=CLASS_NAMES,
            output_dir=OUTPUT_DIR / model_name,
            split_name="test",
        )
        logger.info(
            "[%s] accuracy=%.4f  f1_macro=%.4f",
            model_name,
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )
        return

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = build_loss(cfg["loss"], class_weights=class_weights)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg["optimizer"])
    max_epochs = int(cfg["epochs"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], max_epochs)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    es_cfg = cfg["early_stopping"]
    early_stopping = (
        EarlyStopping(
            monitor=str(es_cfg["monitor"]),
            patience=int(es_cfg["patience"]),
            mode=str(es_cfg["mode"]),
            min_delta=float(es_cfg["min_delta"]),
        )
        if es_cfg["enabled"]
        else None
    )

    ckpt_cfg = cfg["checkpointing"]
    model_checkpoint = (
        ModelCheckpoint(
            dirpath=ctx.checkpoint_dir,
            monitor=str(ckpt_cfg["monitor"]),
            mode=str(ckpt_cfg["mode"]),
            save_top_k=int(ckpt_cfg["save_top_k"]),
            filename_prefix=model_name,
        )
        if ckpt_cfg["enabled"]
        else None
    )

    metric_tracker = MetricTracker()
    callbacks = [cb for cb in [early_stopping, model_checkpoint, metric_tracker] if cb is not None]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=ctx.device,
        num_classes=n_classes,
        max_epochs=max_epochs,
        accumulation_steps=int(cfg["accumulation_steps"]),
        mixed_precision=bool(cfg["mixed_precision"]),
        grad_clip_norm=float(cfg["gradient_clipping"]["max_norm"]),
        callbacks=callbacks,
        scheduler=scheduler,
        wandb_run=None,
        log_every_n_steps=10,
        class_names=CLASS_NAMES,
        checkpoint_dir=ctx.checkpoint_dir,
        monitor_metric="val/f1_macro",
        monitor_mode="max",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training: %d max epochs", max_epochs)
    t_train_start = time.perf_counter()
    history = trainer.fit(train_loader, val_loader)
    train_duration = time.perf_counter() - t_train_start
    logger.info("Training complete in %.1fs (%.1f min)", train_duration, train_duration / 60)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Running test evaluation...")
    test_metrics = trainer.test_epoch(test_loader)
    logger.info(
        "Test: accuracy=%.4f  f1_macro=%.4f  f1_weighted=%.4f",
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
        test_metrics["f1_weighted"],
    )

    # Save test results
    from arst.evaluation.evaluate import _save_evaluation_outputs

    eval_out_dir = OUTPUT_DIR / run_id
    _save_evaluation_outputs(test_metrics, eval_out_dir, "test")

    if model_checkpoint and model_checkpoint.best_model_path:
        logger.info("Best checkpoint: %s", model_checkpoint.best_model_path)

    _, best_val_f1 = metric_tracker.best("val/f1_macro", mode="max")
    logger.info("Best val Macro F1: %.4f", best_val_f1)
    logger.info("Results saved to: %s", eval_out_dir)


if __name__ == "__main__":
    main()
