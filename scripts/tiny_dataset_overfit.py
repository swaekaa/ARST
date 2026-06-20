"""
Phase 2.6 — Tiny Dataset Overfit Sanity Check.

Trains CNN and Transformer on 32 samples for 200 epochs.
Both must reach >95% training accuracy to confirm architectures are functional.

If a model fails to overfit 32 samples, the architecture is fundamentally broken.

Usage::

    python scripts/tiny_dataset_overfit.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arst.data.dataloader import build_csv_loaders
from arst.models.registry import get_model
from arst.training.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tiny_overfit")

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES = 32
N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 32  # = N_SAMPLES, so 1 batch per epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = Path("data/raw/train.csv")

MODELS_TO_TEST = ["cnn", "transformer"]


def get_tiny_batch(csv_path: Path, n_samples: int) -> dict:
    """Load a small subset of data as a single batch."""
    # Use the existing loader infrastructure but with max_rows to get a small set
    train_loader, _, _, data_info = build_csv_loaders(
        csv_path=csv_path,
        window_size=64,
        batch_size=n_samples,  # all in one batch
        val_fraction=0.01,
        test_fraction=0.01,
        num_workers=0,
        seed=42,
        max_rows=5000,  # load enough rows to get sequences
    )

    # Get just the first batch
    batch = next(iter(train_loader))
    # Limit to N_SAMPLES
    tiny = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            tiny[key] = batch[key][:n_samples]
    return tiny, data_info


def overfit_model(model_name: str, batch: dict, n_epochs: int) -> dict:
    """Train a model to overfit a tiny batch. Returns training metrics per epoch."""
    logger.info("=" * 60)
    logger.info(f"OVERFIT TEST: {model_name}")
    logger.info("=" * 60)

    # Build model with NO dropout for overfit test
    model_kwargs = {"num_classes": 4, "active_modalities": ["imu", "thermo", "tof"]}
    if model_name == "cnn":
        model_kwargs.update({
            "imu_channels": 7,
            "thermal_channels": 5,
            "tof_channels": 320,
            "cnn_out_channels": 64,
            "kernel_sizes": [3, 7],
            "tof_proj_dim": 64,
            "head_hidden_dim": 256,
            "dropout": 0.0,  # NO dropout for overfit test
        })
    elif model_name == "transformer":
        model_kwargs.update({
            "imu_channels": 7,
            "thermal_channels": 5,
            "tof_channels": 320,
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 512,
            "dropout": 0.0,  # NO dropout for overfit test
            "pool_type": "cls",
        })

    model = get_model(model_name, **model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model: {model.__class__.__name__} | Params: {n_params:,}")

    # Move batch to device
    imu = batch["imu"].to(DEVICE)
    thermo = batch["thermo"].to(DEVICE)
    tof = batch["tof"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    logger.info(f"  Batch shapes: imu={imu.shape} thermo={thermo.shape} tof={tof.shape}")
    logger.info(f"  Labels: {labels.tolist()}")
    logger.info(f"  Label distribution: {torch.bincount(labels, minlength=4).tolist()}")

    # Simple cross-entropy loss (no focal, no class weights for overfit test)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"loss": [], "accuracy": []}

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(imu=imu, thermo=thermo, tof=tof)
        loss = loss_fn(logits, labels)
        loss.backward()

        # Check gradient health
        if epoch == 0:
            total_grad_norm = 0.0
            zero_grad_layers = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_norm = p.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    if grad_norm < 1e-8:
                        zero_grad_layers.append(name)
            total_grad_norm = total_grad_norm ** 0.5
            logger.info(f"  Epoch 0 — Total grad norm: {total_grad_norm:.6f}")
            if zero_grad_layers:
                logger.warning(f"  Zero-gradient layers: {zero_grad_layers}")

        optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(acc)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            logger.info(
                f"  Epoch {epoch:03d}/{n_epochs}: loss={loss.item():.4f}  acc={acc:.4f}"
            )

        # Check for NaN
        if not torch.isfinite(loss):
            logger.error(f"  NaN/Inf loss at epoch {epoch}!")
            break

    # Final report
    final_acc = history["accuracy"][-1]
    final_loss = history["loss"][-1]
    max_acc = max(history["accuracy"])

    logger.info(f"\n  RESULT: {model_name}")
    logger.info(f"    Final accuracy: {final_acc:.4f}")
    logger.info(f"    Final loss:     {final_loss:.6f}")
    logger.info(f"    Max accuracy:   {max_acc:.4f}")
    logger.info(f"    PASS: {'✅ YES' if max_acc > 0.95 else '❌ NO (architecture is broken!)'}")

    # Inspect final logits
    model.eval()
    with torch.no_grad():
        final_logits = model(imu=imu, thermo=thermo, tof=tof)
    logger.info(f"  Final logits (first 4):\n{final_logits[:4].cpu().numpy()}")
    logger.info(f"  Final preds: {final_logits.argmax(dim=-1).cpu().tolist()}")
    logger.info(f"  True labels: {labels.cpu().tolist()}")

    return {
        "model": model_name,
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "max_accuracy": max_acc,
        "passed": max_acc > 0.95,
        "history": history,
    }


def main():
    logger.info("Phase 2.6 — Tiny Dataset Overfit Sanity Check")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"N_SAMPLES: {N_SAMPLES}, N_EPOCHS: {N_EPOCHS}, LR: {LR}")

    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    # Load tiny batch
    tiny_batch, data_info = get_tiny_batch(CSV_PATH, N_SAMPLES)
    logger.info(f"Loaded tiny batch with {tiny_batch['imu'].shape[0]} samples")

    results = []
    for model_name in MODELS_TO_TEST:
        result = overfit_model(model_name, tiny_batch, N_EPOCHS)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OVERFIT TEST SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        logger.info(
            f"  {r['model']:15s}  max_acc={r['max_accuracy']:.4f}  "
            f"final_acc={r['final_accuracy']:.4f}  {status}"
        )
        if not r["passed"]:
            all_passed = False

    if all_passed:
        logger.info("\n✅ All models can overfit tiny data — architectures are functional.")
    else:
        logger.info("\n❌ Some models CANNOT overfit tiny data — architectures need repair!")

    return results


if __name__ == "__main__":
    main()
