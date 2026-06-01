"""
Full ARST Model — Top-level integration of all components.

Integrates:
  - IMUEncoder, ThermalEncoder, ToFEncoder
  - MultiModalReliabilityModule (ARM per modality)
  - Reliability gating (Hadamard product)
  - AdaptiveFusionTransformer (AFT)
  - ClassificationHead
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from arst.models.encoders.imu_encoder import IMUEncoder
from arst.models.encoders.thermal_encoder import ThermalEncoder
from arst.models.encoders.tof_encoder import ToFEncoder
from arst.models.fusion.adaptive_fusion_transformer import AdaptiveFusionTransformer
from arst.models.heads.classification import ClassificationHead
from arst.models.reliability.arm import MultiModalReliabilityModule


@dataclass
class ARSTOutput:
    """Container for ARST forward pass outputs."""

    logits: torch.Tensor  # [B, C]
    probabilities: torch.Tensor  # [B, C]
    reliability_scores: list[torch.Tensor]  # N × [B, T, 1]
    fused_representation: torch.Tensor  # [B, D]
    attn_bias: torch.Tensor  # [B, S] — for visualization
    embeddings: dict[str, torch.Tensor]  # {"imu": [B,T,D], ...}


class ARSTModel(nn.Module):
    """
    Adaptive Reliability Sensor Transformer — Full Model.

    Args:
        d_model: Shared embedding dimension for all encoders and fusion.
        num_classes: Number of behavior classes.
        imu_config: Config dict for IMUEncoder.
        thermal_config: Config dict for ThermalEncoder.
        tof_config: Config dict for ToFEncoder.
        reliability_config: Config dict for MultiModalReliabilityModule.
        fusion_config: Config dict for AdaptiveFusionTransformer.
        head_config: Config dict for ClassificationHead.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 10,
        imu_config: dict | None = None,
        thermal_config: dict | None = None,
        tof_config: dict | None = None,
        reliability_config: dict | None = None,
        fusion_config: dict | None = None,
        head_config: dict | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Defaults
        imu_config = imu_config or {}
        thermal_config = thermal_config or {}
        tof_config = tof_config or {}
        reliability_config = reliability_config or {}
        fusion_config = fusion_config or {}
        head_config = head_config or {}

        # --- Encoders ---
        self.imu_encoder = IMUEncoder(in_channels=6, d_model=d_model, **imu_config)
        self.thermal_encoder = ThermalEncoder(in_channels=64, d_model=d_model, **thermal_config)
        self.tof_encoder = ToFEncoder(in_channels=64, d_model=d_model, **tof_config)

        # --- Reliability Module ---
        self.reliability_enabled = reliability_config.get("enabled", True)
        if self.reliability_enabled:
            self.reliability_module = MultiModalReliabilityModule(
                n_modalities=3,
                d_model=d_model,
                activation=reliability_config.get("activation", "sigmoid"),
                d_hidden=reliability_config.get("d_hidden", d_model // 4),
                dropout=reliability_config.get("dropout", 0.1),
            )

        # --- Fusion ---
        self.fusion = AdaptiveFusionTransformer(
            d_model=d_model,
            num_layers=fusion_config.get("num_layers", 4),
            num_heads=fusion_config.get("num_heads", 8),
            d_ff=fusion_config.get("d_ff", 1024),
            dropout=fusion_config.get("dropout", 0.1),
            use_reliability_bias=fusion_config.get("use_reliability_bias", True)
            and self.reliability_enabled,
            use_modal_tokens=fusion_config.get("use_modal_tokens", True),
            pool_type=fusion_config.get("pool_type", "cls"),
        )

        # --- Classification Head ---
        self.classification_head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            d_hidden=head_config.get("d_hidden", 256),
            dropout=head_config.get("dropout", 0.1),
        )

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor,
        tof: torch.Tensor,
        tof_mask: torch.Tensor | None = None,
    ) -> ARSTOutput:
        """
        Args:
            imu:      [B, T, 6]   IMU sequence.
            thermo:   [B, T, 64]  Thermopile sequence.
            tof:      [B, T, 64]  ToF sequence.
            tof_mask: [B, T, 64]  Valid pixel mask for ToF (1=valid, 0=invalid).

        Returns:
            ARSTOutput with logits, probabilities, reliability_scores, etc.
        """
        # ── Step 1: Encode each modality ──────────────────────────────────
        h_imu = self.imu_encoder(imu)  # [B, T, D]
        h_thm = self.thermal_encoder(thermo)  # [B, T, D]
        h_tof = self.tof_encoder(tof, tof_mask)  # [B, T, D]

        embeddings = {"imu": h_imu, "thermo": h_thm, "tof": h_tof}

        # ── Step 2: Estimate reliability scores ───────────────────────────
        if self.reliability_enabled:
            reliability_scores = self.reliability_module([h_imu, h_thm, h_tof])
            r_imu, r_thm, r_tof = reliability_scores

            # Reliability-gated embeddings: Ĥ_m = r_m ⊙ H_m
            h_imu_gated = r_imu * h_imu
            h_thm_gated = r_thm * h_thm
            h_tof_gated = r_tof * h_tof
        else:
            reliability_scores = [
                torch.ones(h_imu.shape[0], h_imu.shape[1], 1, device=imu.device) for _ in range(3)
            ]
            h_imu_gated, h_thm_gated, h_tof_gated = h_imu, h_thm, h_tof
            r_imu, r_thm, r_tof = reliability_scores

        # ── Step 3: Adaptive Fusion ────────────────────────────────────────
        fused, attn_bias = self.fusion(
            embeddings=[h_imu_gated, h_thm_gated, h_tof_gated],
            reliability_scores=[r_imu, r_thm, r_tof] if self.reliability_enabled else None,
        )  # fused: [B, D]

        # ── Step 4: Classification ─────────────────────────────────────────
        logits = self.classification_head(fused)  # [B, C]
        probabilities = torch.softmax(logits, dim=-1)  # [B, C]

        return ARSTOutput(
            logits=logits,
            probabilities=probabilities,
            reliability_scores=reliability_scores,
            fused_representation=fused,
            attn_bias=attn_bias,
            embeddings=embeddings,
        )

    def get_reliability_stats(self, output: ARSTOutput) -> dict[str, float]:
        """Extract reliability statistics for W&B logging."""
        names = ["imu", "thermo", "tof"]
        stats = {}
        for name, r in zip(names, output.reliability_scores):
            stats[f"reliability/{name}_mean"] = r.mean().item()
            stats[f"reliability/{name}_std"] = r.std().item()
            stats[f"reliability/{name}_min"] = r.min().item()
            stats[f"reliability/{name}_max"] = r.max().item()
        return stats

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
