"""
Evaluation metrics for ARST Phase 2.

Provides a unified :class:`MetricsCalculator` that accumulates batch predictions
throughout an epoch and computes all metrics at epoch-end via scikit-learn.

Metrics computed:
    - Accuracy
    - Macro F1-Score (primary metric)
    - Weighted F1-Score
    - Per-class F1-Score
    - Confusion matrix (normalized + raw)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

logger = logging.getLogger(__name__)

# Canonical class names from Phase 1 EDA
CLASS_NAMES: list[str] = [
    "Hand at target location",
    "Moves hand to target location",
    "Performs gesture",
    "Relaxes and moves hand to target location",
]


class MetricsCalculator:
    """
    Accumulates per-batch predictions and computes epoch-level metrics.

    Usage::

        calc = MetricsCalculator(num_classes=4)
        for batch in loader:
            logits = model(batch)
            calc.update(logits, batch["label"])
        metrics = calc.compute()
        calc.reset()

    Args:
        num_classes: Number of behavior classes (4 for this dataset).
        class_names: Optional list of class name strings for reporting.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: list[str] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self._preds: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def reset(self) -> None:
        """Clear accumulated predictions for a new epoch."""
        self._preds = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate batch predictions.

        Args:
            logits:  Raw model output [B, num_classes] (unnormalised).
            targets: Ground-truth class indices [B].
        """
        preds = logits.detach().argmax(dim=-1).cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        self._preds.append(preds)
        self._targets.append(targets_np)

    def compute(self) -> dict[str, Any]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary containing:
                - ``accuracy``          : float
                - ``f1_macro``          : float (primary metric)
                - ``f1_weighted``       : float
                - ``f1_per_class``      : dict[str, float]
                - ``confusion_matrix``  : np.ndarray [num_classes, num_classes] (raw counts)
                - ``confusion_matrix_norm``: np.ndarray (row-normalized)
                - ``n_samples``         : int
        """
        if not self._preds:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        all_preds = np.concatenate(self._preds)
        all_targets = np.concatenate(self._targets)
        n = len(all_targets)

        accuracy = float(accuracy_score(all_targets, all_preds))
        f1_macro = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        f1_weighted = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
        f1_per_class_arr = f1_score(
            all_targets,
            all_preds,
            average=None,
            zero_division=0,
            labels=list(range(self.num_classes)),
        )
        f1_per_class = {
            self.class_names[i]: float(f1_per_class_arr[i])
            for i in range(min(len(f1_per_class_arr), len(self.class_names)))
        }

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(self.num_classes)))
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(
                cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0
            )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
            "confusion_matrix_norm": cm_norm,
            "n_samples": n,
        }

    def compute_and_reset(self) -> dict[str, Any]:
        """Compute metrics then reset state. Convenience wrapper."""
        result = self.compute()
        self.reset()
        return result

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def format_metrics(metrics: dict[str, Any], prefix: str = "") -> str:
        """
        Format a metrics dict into a human-readable string for logging.

        Args:
            metrics: Output of :meth:`compute`.
            prefix:  Optional prefix, e.g. ``"val/"`` for namespacing.

        Returns:
            Formatted multi-line string.
        """
        lines = [
            f"{prefix}accuracy    : {metrics['accuracy']:.4f}",
            f"{prefix}f1_macro    : {metrics['f1_macro']:.4f}  (primary)",
            f"{prefix}f1_weighted : {metrics['f1_weighted']:.4f}",
        ]
        lines.append(f"{prefix}per-class F1:")
        for cls_name, f1 in metrics["f1_per_class"].items():
            lines.append(f"  {cls_name[:40]:<40} : {f1:.4f}")
        return "\n".join(lines)
