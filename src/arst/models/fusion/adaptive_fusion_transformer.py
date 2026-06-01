"""
Adaptive Fusion Transformer (AFT).

Fuses reliability-gated multimodal embeddings using a Transformer
with reliability-biased attention.

The key innovation: instead of unmodified self-attention, attention
logits are additively biased by log(reliability_score + ε), so:
  - High reliability → bias ≈ 0 → attention unaffected
  - Low reliability  → bias → -∞ → attention weight → 0

This is differentiable and avoids hard masking.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ReliabilityBiasedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with an additive reliability bias on attention logits.

    The bias matrix is derived from the per-timestep reliability scores:
        bias[i, j] = log(r_j + eps)     (key-side reliability)

    A token attending to an unreliable key receives near-zero weight.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] sequence (S = 3*T for 3 modalities).
            attn_bias: [B, S] reliability-derived additive bias for keys.
                       broadcast to [B, 1, 1, S] so all queries attend equally.
            key_padding_mask: [B, S] boolean mask (True = ignore key position).

        Returns:
            output: [B, S, D]
        """
        B, S, D = x.shape
        H = self.num_heads

        Q = self.q_proj(x)  # [B, S, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to multi-head format
        Q = rearrange(Q, "b s (h d) -> b h s d", h=H)
        K = rearrange(K, "b s (h d) -> b h s d", h=H)
        V = rearrange(V, "b s (h d) -> b h s d", h=H)

        # Scaled dot-product attention logits
        scale = math.sqrt(self.d_head)
        attn_logits = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / scale  # [B, H, S, S]

        # Add reliability bias (broadcast over heads and query positions)
        if attn_bias is not None:
            # attn_bias: [B, S] → [B, 1, 1, S]
            bias = attn_bias.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits + bias

        # Key padding mask
        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted values
        out = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, V)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)

        return out


class AFTBlock(nn.Module):
    """Single AFT Transformer block with reliability-biased attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = ReliabilityBiasedMultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN Transformer (more stable training)
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.ff(self.norm2(x))
        return x


class AdaptiveFusionTransformer(nn.Module):
    """
    Adaptive Fusion Transformer (AFT).

    Fuses three modality embeddings with reliability-biased cross-modal attention.

    Architecture:
      1. Optionally prepend learned [MODAL] token per modality.
      2. Concatenate all modality sequences: [B, 3T, D] (or [B, 3(T+1), D]).
      3. Build reliability bias: log(r + eps) for each key position.
      4. Apply L AFT blocks (reliability-biased Transformer).
      5. Pool: extract CLS token or compute mean.
      → Output: [B, D]

    Args:
        d_model: Embedding dimension.
        num_layers: Number of AFT blocks.
        num_heads: Number of attention heads.
        d_ff: Feed-forward dim.
        dropout: Dropout probability.
        use_reliability_bias: Whether to apply reliability bias.
        use_modal_tokens: Whether to prepend modality-type tokens.
        pool_type: "cls" | "mean".
        n_modalities: Number of sensor modalities.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_reliability_bias: bool = True,
        use_modal_tokens: bool = True,
        pool_type: str = "cls",
        n_modalities: int = 3,
    ):
        super().__init__()
        self.use_reliability_bias = use_reliability_bias
        self.use_modal_tokens = use_modal_tokens
        self.pool_type = pool_type
        self.n_modalities = n_modalities
        self.d_model = d_model

        # Modality-type embeddings (like token-type embeddings in BERT)
        if use_modal_tokens:
            self.modal_embeddings = nn.Embedding(n_modalities, d_model)

        # CLS token for "cls" pooling
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # AFT blocks
        self.blocks = nn.ModuleList(
            [AFTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        reliability_scores: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: List of N tensors, each [B, T, D] (reliability-gated).
            reliability_scores: List of N tensors, each [B, T, 1] in (0, 1).

        Returns:
            fused: [B, D] pooled fused representation.
            attn_bias: [B, S] reliability bias used (for logging/visualization).
        """
        B = embeddings[0].shape[0]
        T = embeddings[0].shape[1]
        device = embeddings[0].device

        # Add modality-type embeddings
        if self.use_modal_tokens:
            modal_ids = torch.arange(self.n_modalities, device=device)
            modal_embs = self.modal_embeddings(modal_ids)  # [N_mod, D]
            embeddings = [
                emb + modal_embs[i].unsqueeze(0).unsqueeze(0) for i, emb in enumerate(embeddings)
            ]

        # Concatenate all modalities: [B, N_mod*T, D]
        x = torch.cat(embeddings, dim=1)  # [B, 3T, D]
        S = x.shape[1]

        # Prepend CLS token
        if self.pool_type == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls, x], dim=1)  # [B, 1+3T, D]
            cls_offset = 1
        else:
            cls_offset = 0

        # Build reliability bias: [B, S] (log-domain)
        attn_bias = None
        if self.use_reliability_bias and reliability_scores is not None:
            # Concatenate reliability scores: [B, 3T, 1]
            r_cat = torch.cat(reliability_scores, dim=1)  # [B, 3T, 1]
            r_cat = r_cat.squeeze(-1)  # [B, 3T]

            # Compute log-domain bias
            log_r = torch.log(r_cat.clamp(min=1e-6))  # [B, 3T]

            if self.pool_type == "cls":
                # Prepend zero bias for CLS token (attend normally)
                cls_bias = torch.zeros(B, 1, device=device)
                log_r = torch.cat([cls_bias, log_r], dim=1)  # [B, 1+3T]

            attn_bias = log_r  # [B, S]

        # Apply AFT blocks
        for block in self.blocks:
            x = block(x, attn_bias=attn_bias)

        x = self.norm(x)

        # Pool
        if self.pool_type == "cls":
            fused = x[:, 0, :]  # CLS token: [B, D]
        else:
            fused = x[:, cls_offset:, :].mean(dim=1)  # Mean pool: [B, D]

        return fused, (attn_bias if attn_bias is not None else torch.zeros(B, S))
