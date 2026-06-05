"""
Unit tests for Phase 2 baseline models.

Tests all 6 baseline models in all 4 modality modes:
    1. IMU-only
    2. Thermal-only
    3. ToF-only
    4. Early-fusion (all 3)

Verifies:
    - Forward pass runs without error
    - Output shape is [B, num_classes]
    - Model accepts tof_mask kwarg gracefully
    - MajorityBaseline.fit() changes predictions
    - Registry instantiation works
"""

from __future__ import annotations

import pytest
import torch

# Phase 1 verified dims
B = 4
T = 64
IMU_CH = 7
THM_CH = 5
TOF_CH = 320
N_CLASSES = 4


def _make_batch(
    active_modalities: list[str],
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Create a synthetic batch for the given active modalities."""
    batch = {
        "imu": torch.randn(B, T, IMU_CH, device=device),
        "thermo": torch.randn(B, T, THM_CH, device=device),
        "tof": torch.randn(B, T, TOF_CH, device=device),
        "tof_mask": (torch.randn(B, T, TOF_CH, device=device) > 0).float(),
        "label": torch.randint(0, N_CLASSES, (B,), device=device),
    }
    # Zero out inactive modalities to detect accidental use
    if "imu" not in active_modalities:
        batch["imu"] = torch.zeros_like(batch["imu"])
    if "thermo" not in active_modalities:
        batch["thermo"] = torch.zeros_like(batch["thermo"])
    if "tof" not in active_modalities:
        batch["tof"] = torch.zeros_like(batch["tof"])
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# MajorityBaseline & RandomBaseline
# ──────────────────────────────────────────────────────────────────────────────


class TestMajorityBaseline:
    def test_output_shape(self):
        from arst.models.baselines.majority import MajorityBaseline

        model = MajorityBaseline(num_classes=N_CLASSES, majority_class=2)
        batch = _make_batch(["imu"])
        logits = model(imu=batch["imu"])
        assert logits.shape == (B, N_CLASSES), f"Expected ({B}, {N_CLASSES}), got {logits.shape}"

    def test_always_predicts_majority(self):
        from arst.models.baselines.majority import MajorityBaseline

        model = MajorityBaseline(num_classes=N_CLASSES, majority_class=3)
        batch = _make_batch(["imu"])
        logits = model(imu=batch["imu"])
        preds = logits.argmax(dim=-1)
        assert (preds == 3).all(), "MajorityBaseline should always predict class 3"

    def test_fit_changes_majority(self):
        from arst.models.baselines.majority import MajorityBaseline

        model = MajorityBaseline(num_classes=N_CLASSES)
        # Class 2 appears most often
        labels = [0, 2, 2, 2, 1, 2, 3]
        model.fit(labels)
        assert model.majority_class == 2


class TestRandomBaseline:
    def test_output_shape(self):
        from arst.models.baselines.majority import RandomBaseline

        model = RandomBaseline(num_classes=N_CLASSES)
        batch = _make_batch(["imu"])
        logits = model(imu=batch["imu"])
        assert logits.shape == (B, N_CLASSES)

    def test_fit_stores_probs(self):

        from arst.models.baselines.majority import RandomBaseline

        model = RandomBaseline(num_classes=N_CLASSES)
        labels = [0, 0, 1, 2, 2, 2, 3]
        model.fit(labels)
        assert abs(model.class_probs.sum() - 1.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# MLP Baseline
# ──────────────────────────────────────────────────────────────────────────────


class TestMLPBaseline:
    @pytest.mark.parametrize(
        "active_modalities",
        [
            ["imu"],
            ["thermo"],
            ["tof"],
            ["imu", "thermo", "tof"],
        ],
    )
    def test_output_shape(self, active_modalities):
        from arst.models.baselines.mlp import MLPBaseline

        model = MLPBaseline(
            num_classes=N_CLASSES,
            active_modalities=active_modalities,
        )
        batch = _make_batch(active_modalities)
        logits = model(
            imu=batch["imu"],
            thermo=batch["thermo"],
            tof=batch["tof"],
            tof_mask=batch["tof_mask"],
        )
        assert logits.shape == (B, N_CLASSES)

    def test_accepts_tof_mask_kwarg(self):
        from arst.models.baselines.mlp import MLPBaseline

        model = MLPBaseline(num_classes=N_CLASSES)
        batch = _make_batch(["imu", "thermo", "tof"])
        # Must not raise even if tof_mask is provided
        logits = model(**{k: v for k, v in batch.items() if k != "label"})
        assert logits.shape == (B, N_CLASSES)

    def test_invalid_modality_raises(self):
        from arst.models.baselines.mlp import MLPBaseline

        with pytest.raises(ValueError):
            MLPBaseline(num_classes=N_CLASSES, active_modalities=[])


# ──────────────────────────────────────────────────────────────────────────────
# CNN Baseline
# ──────────────────────────────────────────────────────────────────────────────


class TestCNNBaseline:
    @pytest.mark.parametrize(
        "active_modalities",
        [
            ["imu"],
            ["thermo"],
            ["tof"],
            ["imu", "thermo", "tof"],
        ],
    )
    def test_output_shape(self, active_modalities):
        from arst.models.baselines.cnn import CNNBaseline

        model = CNNBaseline(num_classes=N_CLASSES, active_modalities=active_modalities)
        batch = _make_batch(active_modalities)
        logits = model(
            imu=batch["imu"],
            thermo=batch["thermo"],
            tof=batch["tof"],
        )
        assert logits.shape == (B, N_CLASSES)

    def test_tof_projection_reduces_params(self):
        from arst.models.baselines.cnn import CNNBaseline

        # tof_proj_dim=32 should have fewer params than tof_proj_dim=64
        model_small = CNNBaseline(num_classes=N_CLASSES, tof_proj_dim=32)
        model_large = CNNBaseline(num_classes=N_CLASSES, tof_proj_dim=128)
        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())
        assert params_small < params_large


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Baseline
# ──────────────────────────────────────────────────────────────────────────────


class TestLSTMBaseline:
    @pytest.mark.parametrize(
        "active_modalities",
        [
            ["imu"],
            ["thermo"],
            ["tof"],
            ["imu", "thermo", "tof"],
        ],
    )
    def test_output_shape(self, active_modalities):
        from arst.models.baselines.lstm import LSTMBaseline

        model = LSTMBaseline(
            num_classes=N_CLASSES,
            active_modalities=active_modalities,
            hidden_size=32,  # small for speed
        )
        batch = _make_batch(active_modalities)
        logits = model(
            imu=batch["imu"],
            thermo=batch["thermo"],
            tof=batch["tof"],
        )
        assert logits.shape == (B, N_CLASSES)

    def test_attention_pool_shape(self):
        from arst.models.baselines.lstm import AttentionPool

        pool = AttentionPool(d_in=64)
        x = torch.randn(B, T, 64)
        out = pool(x)
        assert out.shape == (B, 64)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Baseline
# ──────────────────────────────────────────────────────────────────────────────


class TestTransformerBaseline:
    @pytest.mark.parametrize(
        "active_modalities,pool_type",
        [
            (["imu"], "cls"),
            (["thermo"], "mean"),
            (["tof"], "cls"),
            (["imu", "thermo", "tof"], "cls"),
            (["imu", "thermo", "tof"], "mean"),
        ],
    )
    def test_output_shape(self, active_modalities, pool_type):
        from arst.models.baselines.transformer import TransformerBaseline

        model = TransformerBaseline(
            num_classes=N_CLASSES,
            active_modalities=active_modalities,
            d_model=32,  # tiny for speed
            num_layers=1,
            num_heads=4,
            d_ff=64,
            pool_type=pool_type,
        )
        batch = _make_batch(active_modalities)
        logits = model(
            imu=batch["imu"],
            thermo=batch["thermo"],
            tof=batch["tof"],
        )
        assert logits.shape == (B, N_CLASSES)


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    @pytest.mark.parametrize(
        "model_name", ["majority", "random", "mlp", "cnn", "lstm", "transformer"]
    )
    def test_registry_instantiation(self, model_name):
        from arst.models.registry import get_model

        kwargs: dict = {"num_classes": N_CLASSES}
        if model_name in ("lstm",):
            kwargs["hidden_size"] = 16
        if model_name in ("transformer",):
            kwargs["d_model"] = 32
            kwargs["num_layers"] = 1
            kwargs["num_heads"] = 4

        model = get_model(model_name, **kwargs)
        assert model is not None

    def test_unknown_model_raises(self):
        from arst.models.registry import get_model

        with pytest.raises(KeyError):
            get_model("nonexistent_model")

    def test_list_models(self):
        from arst.models.registry import list_models

        models = list_models()
        assert "mlp" in models
        assert "transformer" in models
        assert "majority" in models
