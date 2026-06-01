"""
Adaptive Reliability Module (ARM).

Estimates per-timestep, per-modality reliability scores from encoder output.
Reliability scores are in (0, 1) and serve as soft gates in the fusion module.

Two activation variants:
  - sigmoid: Independent per-modality; scores can all be high or all low.
  - softmax: Competitive across modalities; scores sum to 1 at each timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveReliabilityModule(nn.Module):
    """
    Lightweight MLP that maps encoder output to a reliability score.

    Architecture:
        H_m [B, T, D]
        → Linear(D, D_h)
        → ReLU
        → Dropout
        → Linear(D_h, 1)
        → [sigmoid | stored as logit for softmax]
        → r_m [B, T, 1]

    Args:
        d_model: Encoder output dimension D.
        d_hidden: Bottleneck hidden dimension (default D//4).
        activation: "sigmoid" or "softmax" (cross-modality — requires stacking).
        dropout: Dropout probability.
        init_bias: Bias for output linear (default 0 → σ(0) = 0.5).
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        activation: str = "sigmoid",
        dropout: float = 0.1,
        init_bias: float = 0.0,
    ):
        super().__init__()
        d_hidden = d_hidden or (d_model // 4)
        self.activation = activation

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, 1),
        )

        # Initialize output bias so initial reliability = 0.5
        nn.init.constant_(self.net[-1].bias, init_bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D] encoder output for one modality.

        Returns:
            r: [B, T, 1] reliability score (or logit if activation='softmax').
                If sigmoid: values in (0, 1).
                If softmax: raw logits — caller must apply softmax across modalities.
        """
        logit = self.net(h)  # [B, T, 1]

        if self.activation == "sigmoid":
            return torch.sigmoid(logit)
        elif self.activation == "softmax":
            # Return raw logits; softmax is applied across modalities in the fusion module
            return logit
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class MultiModalReliabilityModule(nn.Module):
    """
    Wraps N per-modality reliability heads and handles softmax normalization.

    Args:
        n_modalities: Number of sensor modalities (default 3).
        d_model: Encoder embedding dimension.
        activation: "sigmoid" | "softmax".
        d_hidden: ARM hidden dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_modalities: int = 3,
        d_model: int = 256,
        activation: str = "sigmoid",
        d_hidden: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.activation = activation

        self.heads = nn.ModuleList(
            [
                AdaptiveReliabilityModule(
                    d_model=d_model,
                    d_hidden=d_hidden,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(n_modalities)
            ]
        )

    def forward(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Compute reliability scores for all modalities.

        Args:
            embeddings: List of N tensors, each [B, T, D].

        Returns:
            List of N tensors, each [B, T, 1] with reliability in (0, 1).
        """
        assert len(embeddings) == self.n_modalities

        if self.activation == "sigmoid":
            return [head(h) for head, h in zip(self.heads, embeddings)]

        elif self.activation == "softmax":
            # Compute raw logits per modality
            logits = [head(h) for head, h in zip(self.heads, embeddings)]
            # Stack: [B, T, N_mod] → softmax over modality dim → split back
            stacked = torch.cat(logits, dim=-1)  # [B, T, N_mod]
            scores = F.softmax(stacked, dim=-1)  # [B, T, N_mod]
            return [scores[..., i : i + 1] for i in range(self.n_modalities)]

        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class ReliabilityLoss(nn.Module):
    """
    Regularization losses for reliability scores to prevent collapse.

    Components:
        - Entropy regularization: maximize entropy → prevent score collapse to 0/1
        - Diversity regularization: encourage diversity across modalities

    Args:
        lambda_entropy: Weight for entropy regularization.
        lambda_diversity: Weight for diversity regularization.
    """

    def __init__(self, lambda_entropy: float = 0.01, lambda_diversity: float = 0.0):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_diversity = lambda_diversity

    def forward(self, reliability_scores: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            reliability_scores: List of N tensors, each [B, T, 1] in (0, 1).

        Returns:
            dict with "total", "entropy", "diversity" losses.
        """
        losses = {}

        # --- Entropy regularization ---
        # Binary entropy: -[r*log(r) + (1-r)*log(1-r)] — encourage values near 0.5
        if self.lambda_entropy > 0:
            entropy_loss = 0.0
            for r in reliability_scores:
                r_clamped = r.clamp(1e-6, 1 - 1e-6)
                entropy = -(
                    r_clamped * torch.log(r_clamped) + (1 - r_clamped) * torch.log(1 - r_clamped)
                )
                entropy_loss -= entropy.mean()  # negative because we maximize entropy
            losses["entropy"] = self.lambda_entropy * entropy_loss
        else:
            losses["entropy"] = torch.tensor(0.0)

        # --- Diversity regularization ---
        # Penalize high covariance between modality scores (encourage diversity)
        if self.lambda_diversity > 0 and len(reliability_scores) > 1:
            # Stack: [B, T, N_mod]
            stacked = torch.cat(reliability_scores, dim=-1).squeeze()  # [B*T, N_mod]
            if stacked.dim() == 1:
                stacked = stacked.unsqueeze(0)
            cov = torch.cov(stacked.T)  # [N_mod, N_mod]
            # Off-diagonal entries represent cross-modality correlation
            off_diag = cov - torch.diag(torch.diag(cov))
            losses["diversity"] = self.lambda_diversity * off_diag.abs().mean()
        else:
            losses["diversity"] = torch.tensor(0.0)

        losses["total"] = sum(v for k, v in losses.items() if k != "total")
        return losses
