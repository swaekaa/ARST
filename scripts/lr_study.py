"""
Phase 2.6 — Learning Rate Study for CNN and Transformer.

Tests LRs [1e-3, 5e-4, 1e-4, 5e-5] for both CNN and Transformer.
Short runs (20 epochs) to identify optimal learning rate before full training.

Usage::

    python scripts/lr_study.py
"""

from __future__ import annotations

import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arst.data.dataloader import build_csv_loaders
from arst.models.registry import get_model
from arst.training.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lr_study")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = Path("data/raw/train.csv")
N_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATES = [1e-3, 5e-4, 1e-4, 5e-5]
MODELS = ["cnn", "transformer"]

MODEL_CONFIGS = {
    "cnn": {
        "imu_channels": 7,
        "thermal_channels": 5,
        "tof_channels": 320,
        "cnn_out_channels": 64,
        "kernel_sizes": [3, 7],
        "tof_proj_dim": 64,
        "head_hidden_dim": 256,
        "dropout": 0.1,
    },
    "transformer": {
        "imu_channels": 7,
        "thermal_channels": 5,
        "tof_channels": 320,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "pool_type": "cls",
    },
}


def train_with_lr(model_name: str, lr: float, train_loader, val_loader, n_classes: int, class_weights) -> dict:
    """Train a model with a specific LR and return metrics."""
    logger.info(f"  Training {model_name} with LR={lr:.0e}")

    model_kwargs = {
        "num_classes": n_classes,
        "active_modalities": ["imu", "thermo", "tof"],
        **MODEL_CONFIGS[model_name],
    }
    model = get_model(model_name, **model_kwargs).to(DEVICE)

    # Use focal loss with class weights (same as main training)
    from arst.training.losses import build_loss
    loss_fn = build_loss(
        {"cls_type": "focal", "focal": {"gamma": 2.0, "reduction": "mean"}, "use_class_weights": True},
        class_weights=class_weights.to(DEVICE),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_f1 = 0.0
    best_epoch = 0
    train_metrics_calc = MetricsCalculator(num_classes=n_classes)
    val_metrics_calc = MetricsCalculator(num_classes=n_classes)

    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        train_metrics_calc.reset()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imu = batch["imu"].to(DEVICE, non_blocking=True)
            thermo = batch["thermo"].to(DEVICE, non_blocking=True)
            tof = batch["tof"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                logits = model(imu=imu, thermo=thermo, tof=tof)
                loss = loss_fn(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_metrics_calc.update(logits, labels)
            train_loss += loss.item()
            n_batches += 1

        train_computed = train_metrics_calc.compute_and_reset()

        # Validate
        model.eval()
        val_metrics_calc.reset()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                imu = batch["imu"].to(DEVICE, non_blocking=True)
                thermo = batch["thermo"].to(DEVICE, non_blocking=True)
                tof = batch["tof"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)

                with torch.autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                    logits = model(imu=imu, thermo=thermo, tof=tof)
                    loss = loss_fn(logits, labels)

                val_metrics_calc.update(logits, labels)
                val_loss += loss.item()
                val_batches += 1

        val_computed = val_metrics_calc.compute_and_reset()

        if val_computed["f1_macro"] > best_val_f1:
            best_val_f1 = val_computed["f1_macro"]
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            logger.info(
                f"    Epoch {epoch:02d}: train_loss={train_loss/max(n_batches,1):.4f} "
                f"train_f1={train_computed['f1_macro']:.4f} "
                f"val_f1={val_computed['f1_macro']:.4f}"
            )

    return {
        "model": model_name,
        "lr": lr,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
    }


def main():
    logger.info("Phase 2.6 — Learning Rate Study")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"LRs: {LEARNING_RATES}")
    logger.info(f"Models: {MODELS}")
    logger.info(f"Epochs per run: {N_EPOCHS}")

    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    # Load data
    train_loader, val_loader, test_loader, data_info = build_csv_loaders(
        csv_path=CSV_PATH,
        window_size=64,
        batch_size=BATCH_SIZE,
        val_fraction=0.15,
        test_fraction=0.15,
        num_workers=0,
        seed=42,
    )

    results = []
    for model_name in MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"LR STUDY: {model_name}")
        logger.info(f"{'='*60}")

        for lr in LEARNING_RATES:
            result = train_with_lr(
                model_name, lr, train_loader, val_loader,
                data_info["n_classes"], data_info["class_weights"],
            )
            results.append(result)
            logger.info(f"  → Best val F1: {result['best_val_f1']:.4f} at epoch {result['best_epoch']}")

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("LR STUDY RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<15} {'LR':<10} {'Best Val F1':<12} {'Best Epoch':<10}")
    logger.info("-" * 47)
    for r in results:
        logger.info(f"{r['model']:<15} {r['lr']:<10.0e} {r['best_val_f1']:<12.4f} {r['best_epoch']:<10}")

    # Find best LR per model
    for model_name in MODELS:
        model_results = [r for r in results if r["model"] == model_name]
        best = max(model_results, key=lambda x: x["best_val_f1"])
        logger.info(f"\n  Best LR for {model_name}: {best['lr']:.0e} (F1={best['best_val_f1']:.4f})")

    # Save results
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    with open(report_dir / "lr_study_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    report = generate_lr_report(results)
    with open(report_dir / "lr_study.md", "w") as f:
        f.write(report)
    logger.info(f"\nReport saved to {report_dir / 'lr_study.md'}")

    return results


def generate_lr_report(results: list[dict]) -> str:
    """Generate markdown report from LR study results."""
    lines = [
        "# Phase 2.6 — Learning Rate Study",
        "",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> **Epochs per run:** {N_EPOCHS}",
        f"> **Models tested:** {', '.join(MODELS)}",
        "",
        "## Results",
        "",
        "| Model | Learning Rate | Best Val F1 | Best Epoch |",
        "|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['model']} | {r['lr']:.0e} | **{r['best_val_f1']:.4f}** | {r['best_epoch']} |"
        )

    lines.extend(["", "## Best Learning Rate per Model", ""])

    for model_name in MODELS:
        model_results = [r for r in results if r["model"] == model_name]
        best = max(model_results, key=lambda x: x["best_val_f1"])
        lines.append(f"- **{model_name}**: LR={best['lr']:.0e} → F1={best['best_val_f1']:.4f}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
