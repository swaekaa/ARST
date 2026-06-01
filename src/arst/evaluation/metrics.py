"""
Evaluation metrics for ARST behavior recognition.

Wraps scikit-learn and torchmetrics for consistent metric computation
across all experiment phases.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


class BehaviorMetrics:
    """
    Computes standard classification metrics for behavior recognition.

    Usage:
        metrics = BehaviorMetrics(num_classes=10)
        metrics.update(logits, labels)  # call per batch
        results = metrics.compute()     # call at end of epoch
        metrics.reset()
    """

    def __init__(self, num_classes: int, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._all_preds: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._all_probs: list[np.ndarray] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Args:
            logits: [B, C] model output logits.
            labels: [B] ground truth class indices.
        """
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        labels_np = labels.detach().cpu().numpy()

        self._all_preds.append(preds)
        self._all_labels.append(labels_np)
        self._all_probs.append(probs)

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics over accumulated predictions.

        Returns:
            dict with metric names and values.
        """
        all_preds = np.concatenate(self._all_preds)
        all_labels = np.concatenate(self._all_labels)
        all_probs = np.concatenate(self._all_probs)

        results = {}

        # Macro F1 (primary metric)
        results["f1_macro"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Per-class F1
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for c, f1 in enumerate(per_class_f1):
            results[f"f1_class_{c}"] = float(f1)

        # Balanced accuracy
        results["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)

        # AUROC (one-vs-rest, macro)
        try:
            if self.num_classes == 2:
                results["auroc"] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                results["auroc"] = roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class="ovr",
                    average="macro",
                )
        except ValueError:
            results["auroc"] = float("nan")

        # Accuracy
        results["accuracy"] = float((all_preds == all_labels).mean())

        # Confusion matrix (store as nested list for W&B logging)
        results["confusion_matrix"] = confusion_matrix(
            all_labels, all_preds, normalize="true"
        ).tolist()

        return results

    def compute_reliability_correlation(
        self,
        reliability_scores_list: list[np.ndarray],
        modality_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute correlation between aggregate reliability scores and prediction correctness.

        Args:
            reliability_scores_list: List of [N] arrays (mean reliability per sample per modality).
            modality_names: Names for each modality.

        Returns:
            dict of "reliability_accuracy_corr_{modality}" values.
        """
        all_preds = np.concatenate(self._all_preds)
        all_labels = np.concatenate(self._all_labels)
        correct = (all_preds == all_labels).astype(float)

        results = {}
        names = modality_names or [f"modality_{i}" for i in range(len(reliability_scores_list))]

        for name, r in zip(names, reliability_scores_list):
            if len(r) == len(correct):
                corr = float(np.corrcoef(r, correct)[0, 1])
                results[f"reliability_accuracy_corr_{name}"] = corr

        return results
