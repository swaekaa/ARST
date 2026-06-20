"""
Phase 2.6 — Architecture Ablation Study.

Compares:
  Transformer: mean pooling vs CLS token vs attention pooling
  CNN: kernel 3 vs kernel 5 vs multi-scale (3,5,7)

Usage::

    python scripts/ablation_study.py
"""

from __future__ import annotations

import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arst.data.dataloader import build_csv_loaders
from arst.models.registry import get_model
from arst.training.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablation")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = Path("data/raw/train.csv")
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4


def train_and_eval(model, train_loader, val_loader, n_classes, class_weights, tag: str) -> dict:
    """Train a model and return val metrics."""
    logger.info(f"  Training ablation: {tag}")
    model = model.to(DEVICE)

    from arst.training.losses import build_loss
    loss_fn = build_loss(
        {"cls_type": "focal", "focal": {"gamma": 2.0, "reduction": "mean"}, "use_class_weights": True},
        class_weights=class_weights.to(DEVICE),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_val_f1 = 0.0
    val_metrics_calc = MetricsCalculator(num_classes=n_classes)
    train_metrics_calc = MetricsCalculator(num_classes=n_classes)

    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        train_metrics_calc.reset()
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

        train_computed = train_metrics_calc.compute_and_reset()

        # Validate
        model.eval()
        val_metrics_calc.reset()
        with torch.no_grad():
            for batch in val_loader:
                imu = batch["imu"].to(DEVICE, non_blocking=True)
                thermo = batch["thermo"].to(DEVICE, non_blocking=True)
                tof = batch["tof"].to(DEVICE, non_blocking=True)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, enabled=torch.cuda.is_available()):
                    logits = model(imu=imu, thermo=thermo, tof=tof)
                val_metrics_calc.update(logits, labels)

        val_computed = val_metrics_calc.compute_and_reset()
        if val_computed["f1_macro"] > best_val_f1:
            best_val_f1 = val_computed["f1_macro"]

        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            logger.info(
                f"    [{tag}] Epoch {epoch:02d}: "
                f"train_f1={train_computed['f1_macro']:.4f} val_f1={val_computed['f1_macro']:.4f}"
            )

    logger.info(f"  → [{tag}] Best val F1: {best_val_f1:.4f}")
    return {"tag": tag, "best_val_f1": best_val_f1}


def run_transformer_ablations(train_loader, val_loader, n_classes, class_weights):
    """Compare Transformer pooling strategies."""
    results = []
    base_kwargs = {
        "num_classes": n_classes,
        "active_modalities": ["imu", "thermo", "tof"],
        "imu_channels": 7,
        "thermal_channels": 5,
        "tof_channels": 320,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
    }

    # CLS pooling
    model = get_model("transformer", pool_type="cls", **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "transformer_cls"))

    # Mean pooling
    model = get_model("transformer", pool_type="mean", **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "transformer_mean"))

    return results


def run_cnn_ablations(train_loader, val_loader, n_classes, class_weights):
    """Compare CNN kernel configurations."""
    results = []
    base_kwargs = {
        "num_classes": n_classes,
        "active_modalities": ["imu", "thermo", "tof"],
        "imu_channels": 7,
        "thermal_channels": 5,
        "tof_channels": 320,
        "cnn_out_channels": 64,
        "tof_proj_dim": 64,
        "head_hidden_dim": 256,
        "dropout": 0.1,
    }

    # Kernel 3 only
    model = get_model("cnn", kernel_sizes=[3], **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "cnn_k3"))

    # Kernel 5 only
    model = get_model("cnn", kernel_sizes=[5], **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "cnn_k5"))

    # Multi-scale (3, 5, 7)
    model = get_model("cnn", kernel_sizes=[3, 5, 7], **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "cnn_k3_5_7"))

    # Default (3, 7)
    model = get_model("cnn", kernel_sizes=[3, 7], **base_kwargs)
    results.append(train_and_eval(model, train_loader, val_loader, n_classes, class_weights, "cnn_k3_7"))

    return results


def generate_report(all_results: list[dict]) -> str:
    """Generate markdown ablation report."""
    lines = [
        "# Phase 2.6 — Baseline Architecture Ablations",
        "",
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> **Epochs per run:** {N_EPOCHS}",
        f"> **Learning rate:** {LR}",
        "",
        "## Transformer Pooling Ablation",
        "",
        "| Variant | Pool Type | Best Val F1 |",
        "|---|---|---|",
    ]

    transformer_results = [r for r in all_results if r["tag"].startswith("transformer")]
    for r in sorted(transformer_results, key=lambda x: x["best_val_f1"], reverse=True):
        pool_type = r["tag"].replace("transformer_", "")
        lines.append(f"| {r['tag']} | {pool_type} | **{r['best_val_f1']:.4f}** |")

    lines.extend([
        "",
        "## CNN Kernel Ablation",
        "",
        "| Variant | Kernels | Best Val F1 |",
        "|---|---|---|",
    ])

    cnn_results = [r for r in all_results if r["tag"].startswith("cnn")]
    for r in sorted(cnn_results, key=lambda x: x["best_val_f1"], reverse=True):
        kernel = r["tag"].replace("cnn_", "")
        lines.append(f"| {r['tag']} | {kernel} | **{r['best_val_f1']:.4f}** |")

    return "\n".join(lines) + "\n"


def main():
    logger.info("Phase 2.6 — Architecture Ablation Study")
    logger.info(f"Device: {DEVICE}")

    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    train_loader, val_loader, _, data_info = build_csv_loaders(
        csv_path=CSV_PATH,
        window_size=64,
        batch_size=BATCH_SIZE,
        val_fraction=0.15,
        test_fraction=0.15,
        num_workers=0,
        seed=42,
    )

    all_results = []

    logger.info(f"\n{'='*60}")
    logger.info("TRANSFORMER POOLING ABLATION")
    logger.info(f"{'='*60}")
    all_results.extend(run_transformer_ablations(
        train_loader, val_loader, data_info["n_classes"], data_info["class_weights"]
    ))

    logger.info(f"\n{'='*60}")
    logger.info("CNN KERNEL ABLATION")
    logger.info(f"{'='*60}")
    all_results.extend(run_cnn_ablations(
        train_loader, val_loader, data_info["n_classes"], data_info["class_weights"]
    ))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True):
        logger.info(f"  {r['tag']:<25s} F1={r['best_val_f1']:.4f}")

    # Save
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    with open(report_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    report = generate_report(all_results)
    with open(report_dir / "baseline_architecture_ablations.md", "w") as f:
        f.write(report)
    logger.info(f"\nReport saved to {report_dir / 'baseline_architecture_ablations.md'}")

    return all_results


if __name__ == "__main__":
    main()
