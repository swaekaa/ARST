"""
Focal Loss and combined ARST training loss.

FocalLoss addresses class imbalance by down-weighting well-classified examples
(γ > 0), focusing training on hard examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Softmax variant).

    L_focal = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter (γ=0 → standard CE; γ=2 recommended).
        alpha: Per-class weight tensor [C]. If None, uniform weights.
        reduction: "mean" | "sum" | "none".
        label_smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized scores.
            targets: [B] class indices (long).

        Returns:
            Focal loss scalar.
        """
        B, C = logits.shape

        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]

        # Gather p_t for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ARSTLoss(nn.Module):
    """
    Combined ARST training loss:
        L_total = L_cls + λ_rel * L_rel

    Args:
        num_classes: Number of behavior classes.
        cls_type: "focal" | "cross_entropy".
        focal_gamma: Focal loss gamma.
        class_weights: Per-class weights for imbalance handling.
        lambda_rel: Reliability regularization weight.
        lambda_ent: Entropy component weight within L_rel.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_type: str = "focal",
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        lambda_rel: float = 0.1,
        lambda_ent: float = 0.01,
    ):
        super().__init__()
        self.cls_type = cls_type
        self.lambda_rel = lambda_rel

        # Classification loss
        if cls_type == "focal":
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        elif cls_type == "cross_entropy":
            self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unknown cls_type: {cls_type}")

        # Reliability loss
        from arst.models.reliability.arm import ReliabilityLoss

        self.rel_loss = ReliabilityLoss(lambda_entropy=lambda_ent)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reliability_scores: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, C] model output.
            targets: [B] ground-truth class indices.
            reliability_scores: Optional list of N × [B, T, 1] tensors.

        Returns:
            dict with keys: "total", "cls", "rel", "rel_entropy", "rel_diversity".
        """
        losses = {}

        # Classification loss
        losses["cls"] = self.cls_loss(logits, targets)

        # Reliability regularization
        if reliability_scores is not None and self.lambda_rel > 0:
            rel_losses = self.rel_loss(reliability_scores)
            losses["rel_entropy"] = rel_losses["entropy"]
            losses["rel_diversity"] = rel_losses["diversity"]
            losses["rel"] = rel_losses["total"]
        else:
            losses["rel"] = torch.tensor(0.0, device=logits.device)
            losses["rel_entropy"] = torch.tensor(0.0, device=logits.device)
            losses["rel_diversity"] = torch.tensor(0.0, device=logits.device)

        losses["total"] = losses["cls"] + self.lambda_rel * losses["rel"]

        return losses
