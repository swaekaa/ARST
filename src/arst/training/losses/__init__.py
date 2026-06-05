"""
Training losses package for ARST.

Provides:
    - :class:`FocalLoss`: Focal loss for class-imbalanced multi-class classification.
    - :func:`build_loss`: Factory that returns the configured loss from a dict/OmegaConf config.
    - :class:`ARSTLoss`: Combined classification + reliability loss (Phase 4+).

Phase 1 justification:
    The dataset has a 3.79× class imbalance (worst: "Performs gesture" 44.5%
    vs "Relaxes and moves..." 11.7%). Focal Loss with γ=2.0 down-weights
    easy majority-class examples during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Reduces the contribution of well-classified (easy) examples so the model
    focuses on hard examples — particularly useful for the 3.79× imbalance
    confirmed in Phase 1.

    Args:
        gamma: Focusing parameter. γ=0 → standard CE; γ=2 recommended.
        weight: Per-class weights tensor [num_classes]. Pass class_weights
            from the DataModule for additional imbalance correction.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
        label_smoothing: Label-smoothing coefficient (0 = off).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight: torch.Tensor | None = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw model output (unnormalised).
            targets: [B]    ground-truth class indices.

        Returns:
            Scalar loss tensor.
        """
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            p_t = torch.exp(-ce)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_cfg: dict | object,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Factory function: build the configured loss module.

    Args:
        loss_cfg: Dict or OmegaConf DictConfig with at least ``cls_type`` key.
            Supported ``cls_type`` values:

            - ``"focal"``          — :class:`FocalLoss`
            - ``"cross_entropy"``  — :class:`torch.nn.CrossEntropyLoss`

        class_weights: Optional inverse-frequency class weight tensor.
            Injected when ``loss_cfg.use_class_weights`` is ``True``.

    Returns:
        Configured loss module.
    """
    try:
        from omegaconf import OmegaConf

        if isinstance(loss_cfg, object) and hasattr(loss_cfg, "_metadata"):
            loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)
    except ImportError:
        pass

    if isinstance(loss_cfg, dict):
        cls_type = loss_cfg.get("cls_type", "focal")
        use_class_weights: bool = loss_cfg.get("use_class_weights", True)
        focal_cfg: dict = loss_cfg.get("focal", {})
        gamma: float = focal_cfg.get("gamma", 2.0)
        label_smoothing: float = loss_cfg.get("label_smoothing", 0.0)
    else:
        cls_type = getattr(loss_cfg, "cls_type", "focal")
        use_class_weights = getattr(loss_cfg, "use_class_weights", True)
        focal_cfg_obj = getattr(loss_cfg, "focal", None)
        gamma = getattr(focal_cfg_obj, "gamma", 2.0) if focal_cfg_obj else 2.0
        label_smoothing = getattr(loss_cfg, "label_smoothing", 0.0)

    weight = class_weights if use_class_weights and class_weights is not None else None

    if cls_type == "focal":
        return FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)
    elif cls_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {cls_type!r}. Choose 'focal' or 'cross_entropy'.")


__all__ = ["FocalLoss", "build_loss"]
