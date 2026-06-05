"""
Majority and Random baselines for ARST Phase 2.

These are non-trainable sanity-check baselines:
    - :class:`MajorityBaseline`: Always predicts the majority class.
    - :class:`RandomBaseline`:   Samples from the observed class distribution.

Use them to verify that the training loop, metrics, and dataloaders
are all correct before running any learned model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MajorityBaseline(nn.Module):
    """
    Always predicts the most frequent class observed in training data.

    This is a non-trainable model — calling :meth:`fit` stores the
    majority class; thereafter :meth:`forward` returns a one-hot-like
    logit vector with a large score for the majority class.

    Args:
        num_classes: Number of behavior classes (4 for this dataset).
        majority_class: Index of the majority class.  Set via :meth:`fit`
            or pass directly at construction time.
    """

    def __init__(self, num_classes: int = 4, majority_class: int = 0, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.majority_class = majority_class

    def fit(self, labels: list[int] | np.ndarray | torch.Tensor) -> MajorityBaseline:
        """
        Determine majority class from a label sequence.

        Args:
            labels: Array of integer class labels from the training set.

        Returns:
            Self (for chaining).
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels_arr = np.asarray(labels, dtype=int)
        counts = np.bincount(labels_arr, minlength=self.num_classes)
        self.majority_class = int(np.argmax(counts))
        return self

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor | None = None,
        tof: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Return logits that deterministically argmax to ``majority_class``.

        Args:
            imu:    [B, T, C_imu]  — used only to infer batch size.
            thermo: Ignored.
            tof:    Ignored.

        Returns:
            Logits [B, num_classes] with majority class score = 10.0,
            all others = 0.0.
        """
        B = imu.shape[0]
        logits = torch.zeros(B, self.num_classes, device=imu.device)
        logits[:, self.majority_class] = 10.0
        return logits


class RandomBaseline(nn.Module):
    """
    Samples class predictions from the training class distribution.

    Not a learned model — :meth:`fit` records the class probability
    vector; :meth:`forward` returns logits sampled from that distribution
    (with small random perturbations for stochasticity).

    Args:
        num_classes:   Number of behavior classes.
        class_probs:   Initial class probability vector (uniform if ``None``).
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_probs: list[float] | np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if class_probs is None:
            self.class_probs = np.ones(num_classes) / num_classes
        else:
            arr = np.asarray(class_probs, dtype=float)
            self.class_probs = arr / arr.sum()  # normalise

    def fit(self, labels: list[int] | np.ndarray | torch.Tensor) -> RandomBaseline:
        """
        Estimate class distribution from training labels.

        Args:
            labels: Integer label array from the training set.

        Returns:
            Self (for chaining).
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels_arr = np.asarray(labels, dtype=int)
        counts = np.bincount(labels_arr, minlength=self.num_classes).astype(float)
        self.class_probs = counts / counts.sum()
        return self

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor | None = None,
        tof: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Return logits drawn from the learned class distribution.

        Args:
            imu:    [B, T, C_imu]  — used only to infer batch size and device.
            thermo: Ignored.
            tof:    Ignored.

        Returns:
            Logits [B, num_classes] — log-probability + noise.
        """
        B = imu.shape[0]
        # Use log-prob as base logit, add noise for diversity
        log_probs = np.log(self.class_probs + 1e-8)
        logits_np = log_probs + np.random.randn(B, self.num_classes) * 0.1
        return torch.tensor(logits_np, dtype=torch.float32, device=imu.device)
