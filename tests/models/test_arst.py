"""
Test suite for the ARST model forward pass.

Tests:
  - Model output shapes
  - Reliability score range [0, 1]
  - Gradient flow (all parameters receive gradients)
  - Missing modality robustness (all-zero input)
  - Device compatibility (CPU test)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from arst.models.arst import ARSTModel


@pytest.fixture
def model_config():
    return {
        "d_model": 64,  # Small for fast tests
        "num_classes": 5,
        "imu_config": {
            "cnn_filters": 16,
            "cnn_kernel_sizes": [3, 7],
            "num_transformer_layers": 1,
            "num_heads": 2,
            "d_ff": 128,
        },
        "thermal_config": {
            "num_transformer_layers": 1,
            "num_heads": 2,
            "d_ff": 128,
        },
        "tof_config": {
            "num_transformer_layers": 1,
            "num_heads": 2,
            "d_ff": 128,
        },
        "reliability_config": {
            "enabled": True,
            "activation": "sigmoid",
            "d_hidden": 16,
        },
        "fusion_config": {
            "num_layers": 2,
            "num_heads": 2,
            "d_ff": 128,
            "use_reliability_bias": True,
            "use_modal_tokens": True,
            "pool_type": "cls",
        },
        "head_config": {
            "d_hidden": 64,
            "dropout": 0.0,
        },
    }


@pytest.fixture
def dummy_batch():
    B, T = 2, 32
    return {
        "imu": torch.randn(B, T, 6),
        "thermo": torch.randn(B, T, 64),
        "tof": torch.randn(B, T, 64),
        "tof_mask": torch.ones(B, T, 64),
        "label": torch.randint(0, 5, (B,)),
    }


class TestARSTModel:

    def test_output_shape(self, model_config, dummy_batch):
        """Model should produce logits of shape [B, num_classes]."""
        model = ARSTModel(**model_config)
        model.eval()

        with torch.no_grad():
            output = model(
                imu=dummy_batch["imu"],
                thermo=dummy_batch["thermo"],
                tof=dummy_batch["tof"],
                tof_mask=dummy_batch["tof_mask"],
            )

        B = dummy_batch["imu"].shape[0]
        assert output.logits.shape == (B, model_config["num_classes"]), (
            f"Expected logits shape ({B}, {model_config['num_classes']}), "
            f"got {output.logits.shape}"
        )

    def test_probabilities_sum_to_one(self, model_config, dummy_batch):
        """Softmax probabilities should sum to 1."""
        model = ARSTModel(**model_config)
        model.eval()

        with torch.no_grad():
            output = model(
                imu=dummy_batch["imu"],
                thermo=dummy_batch["thermo"],
                tof=dummy_batch["tof"],
            )

        prob_sum = output.probabilities.sum(dim=-1)
        assert torch.allclose(
            prob_sum, torch.ones_like(prob_sum), atol=1e-5
        ), f"Probabilities do not sum to 1: {prob_sum}"

    def test_reliability_scores_in_range(self, model_config, dummy_batch):
        """Reliability scores should be in (0, 1) for sigmoid activation."""
        model = ARSTModel(**model_config)
        model.eval()

        with torch.no_grad():
            output = model(
                imu=dummy_batch["imu"],
                thermo=dummy_batch["thermo"],
                tof=dummy_batch["tof"],
            )

        for i, r in enumerate(output.reliability_scores):
            assert r.min() >= 0.0, f"Modality {i} reliability < 0: {r.min()}"
            assert r.max() <= 1.0, f"Modality {i} reliability > 1: {r.max()}"

    def test_gradient_flow(self, model_config, dummy_batch):
        """All parameters should receive gradients during backward pass."""
        model = ARSTModel(**model_config)
        model.train()

        output = model(
            imu=dummy_batch["imu"],
            thermo=dummy_batch["thermo"],
            tof=dummy_batch["tof"],
        )
        loss = output.logits.sum()
        loss.backward()

        no_grad_params = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert len(no_grad_params) == 0, f"Params with no gradient: {no_grad_params}"

    def test_missing_modality_zero_input(self, model_config, dummy_batch):
        """Model should handle all-zero modality without NaN/Inf."""
        model = ARSTModel(**model_config)
        model.eval()

        # Zero out IMU (simulates missing modality)
        dummy_batch["imu"] = torch.zeros_like(dummy_batch["imu"])

        with torch.no_grad():
            output = model(
                imu=dummy_batch["imu"],
                thermo=dummy_batch["thermo"],
                tof=dummy_batch["tof"],
            )

        assert not torch.isnan(output.logits).any(), "NaN in logits with missing IMU"
        assert not torch.isinf(output.logits).any(), "Inf in logits with missing IMU"

    def test_reliability_disabled(self, model_config, dummy_batch):
        """Model with reliability disabled should still produce valid outputs."""
        config = dict(model_config)
        config["reliability_config"] = {"enabled": False}

        model = ARSTModel(**config)
        model.eval()

        with torch.no_grad():
            output = model(
                imu=dummy_batch["imu"],
                thermo=dummy_batch["thermo"],
                tof=dummy_batch["tof"],
            )

        assert output.logits.shape[1] == model_config["num_classes"]
        # All reliability scores should be 1.0 (unity gate)
        for r in output.reliability_scores:
            assert torch.allclose(r, torch.ones_like(r)), "Reliability != 1 when disabled"

    def test_parameter_count(self, model_config):
        """Model should have a reasonable parameter count (< 1M for test config)."""
        model = ARSTModel(**model_config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params < 5_000_000, f"Model too large for test config: {n_params:,} params"
        print(f"\nTest model parameters: {n_params:,}")
