"""
Unit tests for Phase 2 training metrics.

Verifies:
    - MetricsCalculator accumulates predictions correctly
    - compute() returns correct shapes and types
    - Edge cases: perfect predictions, all-wrong, empty state
    - FocalLoss output shapes and gradient flow
    - build_loss factory creates correct loss types
"""

from __future__ import annotations

import pytest
import torch

# ──────────────────────────────────────────────────────────────────────────────
# MetricsCalculator tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMetricsCalculator:
    def test_perfect_predictions(self):
        from arst.training.metrics import MetricsCalculator

        calc = MetricsCalculator(num_classes=4)
        # Perfect predictions: logits that argmax to the correct class
        for _ in range(5):
            labels = torch.tensor([0, 1, 2, 3])
            logits = torch.eye(4)  # diagonal → argmax = class index
            calc.update(logits, labels)

        metrics = calc.compute()
        assert abs(metrics["accuracy"] - 1.0) < 1e-6
        assert abs(metrics["f1_macro"] - 1.0) < 1e-6
        assert abs(metrics["f1_weighted"] - 1.0) < 1e-6

    def test_all_wrong_predictions(self):
        from arst.training.metrics import MetricsCalculator

        calc = MetricsCalculator(num_classes=4)
        labels = torch.tensor([0, 0, 0, 0])  # all class 0
        # Predict class 1 for everything
        logits = torch.zeros(4, 4)
        logits[:, 1] = 10.0
        calc.update(logits, labels)

        metrics = calc.compute()
        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        from arst.training.metrics import MetricsCalculator

        n_classes = 4
        calc = MetricsCalculator(num_classes=n_classes)
        labels = torch.randint(0, n_classes, (32,))
        logits = torch.randn(32, n_classes)
        calc.update(logits, labels)

        metrics = calc.compute()
        assert metrics["confusion_matrix"].shape == (n_classes, n_classes)
        assert metrics["confusion_matrix_norm"].shape == (n_classes, n_classes)

    def test_reset_clears_state(self):
        from arst.training.metrics import MetricsCalculator

        calc = MetricsCalculator(num_classes=4)
        calc.update(torch.randn(8, 4), torch.randint(0, 4, (8,)))
        calc.reset()
        with pytest.raises(RuntimeError):
            calc.compute()

    def test_compute_and_reset(self):
        from arst.training.metrics import MetricsCalculator

        calc = MetricsCalculator(num_classes=4)
        calc.update(torch.eye(4), torch.arange(4))
        result = calc.compute_and_reset()
        assert "f1_macro" in result
        # After reset, should fail
        with pytest.raises(RuntimeError):
            calc.compute()

    def test_n_samples_correct(self):
        from arst.training.metrics import MetricsCalculator

        calc = MetricsCalculator(num_classes=4)
        calc.update(torch.randn(10, 4), torch.randint(0, 4, (10,)))
        calc.update(torch.randn(20, 4), torch.randint(0, 4, (20,)))
        metrics = calc.compute()
        assert metrics["n_samples"] == 30

    def test_per_class_f1_keys(self):
        from arst.training.metrics import MetricsCalculator

        class_names = ["ClassA", "ClassB", "ClassC", "ClassD"]
        calc = MetricsCalculator(num_classes=4, class_names=class_names)
        calc.update(torch.randn(8, 4), torch.randint(0, 4, (8,)))
        metrics = calc.compute()
        assert set(metrics["f1_per_class"].keys()) == set(class_names)


# ──────────────────────────────────────────────────────────────────────────────
# FocalLoss tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFocalLoss:
    def test_output_shape_scalar(self):
        from arst.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 4)
        labels = torch.randint(0, 4, (16,))
        loss = loss_fn(logits, labels)
        assert loss.shape == ()  # scalar

    def test_gradient_flows(self):
        from arst.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4, requires_grad=True)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_class_weights_applied(self):
        from arst.training.losses import FocalLoss

        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss_fn_weighted = FocalLoss(gamma=0.0, weight=weights)
        loss_fn_uniform = FocalLoss(gamma=0.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        # Losses should differ when class weights are applied
        loss_w = loss_fn_weighted(logits, labels).item()
        loss_u = loss_fn_uniform(logits, labels).item()
        assert loss_w != loss_u

    def test_gamma_zero_equals_cross_entropy(self):
        import torch.nn.functional as F

        from arst.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=0.0)
        logits = torch.randn(16, 4)
        labels = torch.randint(0, 4, (16,))
        focal_loss = loss_fn(logits, labels).item()
        ce_loss = F.cross_entropy(logits, labels).item()
        assert abs(focal_loss - ce_loss) < 1e-5


class TestBuildLoss:
    def test_focal_type(self):
        from arst.training.losses import FocalLoss, build_loss

        cfg = {"cls_type": "focal", "use_class_weights": False, "focal": {"gamma": 2.0}}
        loss = build_loss(cfg)
        assert isinstance(loss, FocalLoss)

    def test_ce_type(self):
        import torch.nn as nn

        from arst.training.losses import build_loss

        cfg = {"cls_type": "cross_entropy", "use_class_weights": False}
        loss = build_loss(cfg)
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_unknown_type_raises(self):
        from arst.training.losses import build_loss

        cfg = {"cls_type": "unknown_loss_type"}
        with pytest.raises(ValueError):
            build_loss(cfg)
