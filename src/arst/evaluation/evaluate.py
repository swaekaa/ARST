"""
Standalone evaluation framework for ARST Phase 2.

Provides :func:`evaluate_model` for running a full evaluation pass
on any split (train/val/test) and saving results to disk.

Outputs written to ``outputs/evaluation/<model_name>/``:
    - ``metrics.json``         — all scalar metrics
    - ``confusion_matrix.npy`` — raw confusion matrix
    - ``confusion_matrix.png`` — visualisation (if matplotlib available)
    - ``per_class_f1.csv``     — per-class F1 table

Usage::

    python -m arst.evaluation.evaluate \\
        --checkpoint experiments/mlp_seed42/checkpoints/model_best.pt \\
        --model mlp \\
        --split test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arst.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    class_names: list[str] | None = None,
    mixed_precision: bool = True,
    output_dir: str | Path | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    """
    Run full evaluation on a DataLoader and return metrics.

    Args:
        model:           The model to evaluate (any :class:`~torch.nn.Module`).
        loader:          DataLoader for the split to evaluate.
        device:          Computation device.
        num_classes:     Number of behavior classes.
        class_names:     Optional list of class name strings.
        mixed_precision: Use AMP during evaluation (faster on GPU).
        output_dir:      If provided, save results here.  ``None`` → no saving.
        split_name:      Label for logging (``"val"`` / ``"test"``).

    Returns:
        Metrics dict from :class:`~arst.training.metrics.MetricsCalculator`
        plus ``"loss"`` key if a loss function is available.
    """
    model.eval()
    calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]", dynamic_ncols=True):
            imu = batch["imu"].to(device, non_blocking=True)
            thermo = batch["thermo"].to(device, non_blocking=True)
            tof = batch["tof"].to(device, non_blocking=True)
            tof_mask = batch.get("tof_mask")
            if tof_mask is not None:
                tof_mask = tof_mask.to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(imu=imu, thermo=thermo, tof=tof, tof_mask=tof_mask)

            calc.update(logits, labels)

    metrics = calc.compute()

    # Log to console
    logger.info(
        "[%s] accuracy=%.4f  f1_macro=%.4f  f1_weighted=%.4f  (n=%d)",
        split_name,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["n_samples"],
    )
    for cls_name, f1 in metrics["f1_per_class"].items():
        logger.info("  %-45s : %.4f", cls_name, f1)

    # Save outputs
    if output_dir is not None:
        _save_evaluation_outputs(metrics, Path(output_dir), split_name)

    return metrics


def _save_evaluation_outputs(
    metrics: dict[str, Any],
    output_dir: Path,
    split_name: str,
) -> None:
    """Persist evaluation results to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scalar metrics → JSON
    scalar_metrics: dict[str, Any] = {
        k: v for k, v in metrics.items() if not isinstance(v, np.ndarray | dict)
    }
    scalar_metrics.update(metrics.get("f1_per_class", {}))

    json_path = output_dir / f"{split_name}_metrics.json"
    with json_path.open("w") as f:
        json.dump(scalar_metrics, f, indent=2)
    logger.info("Saved scalar metrics -> %s", json_path)

    # Confusion matrix → NPY
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        cm_path = output_dir / f"{split_name}_confusion_matrix.npy"
        np.save(cm_path, cm)
        logger.info("Saved confusion matrix -> %s", cm_path)

        # Visualisation (optional)
        _save_confusion_matrix_plot(cm, output_dir, split_name, metrics.get("class_names"))

    # Per-class F1 → CSV
    f1_per_class = metrics.get("f1_per_class", {})
    if f1_per_class:
        csv_path = output_dir / f"{split_name}_per_class_f1.csv"
        with csv_path.open("w") as f:
            f.write("class,f1\n")
            for cls_name, f1 in f1_per_class.items():
                f.write(f'"{cls_name}",{f1:.6f}\n')
        logger.info("Saved per-class F1 -> %s", csv_path)


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    output_dir: Path,
    split_name: str,
    class_names: list[str] | None,
) -> None:
    """Save a normalised confusion matrix heatmap to PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.debug("matplotlib/seaborn not available -- skipping confusion matrix plot.")
        return

    n = cm.shape[0]
    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]

    # Row-normalise
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix ({split_name}, normalised)", fontsize=12)
    plt.tight_layout()

    plot_path = output_dir / f"{split_name}_confusion_matrix.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot -> %s", plot_path)
