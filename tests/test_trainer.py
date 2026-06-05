"""
Trainer smoke test for ARST Phase 2.

Runs a minimal 2-epoch training loop using synthetic data to verify:
    - Trainer initialises without error
    - train_epoch() and validate_epoch() produce correct metric dicts
    - Callbacks are called at expected times
    - Checkpoints are saved to disk
    - test_epoch() returns full metrics dict with confusion matrix

Does NOT require the actual dataset — uses tiny random tensors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Minimal dummy dataset that matches the Trainer's expected batch format
# ──────────────────────────────────────────────────────────────────────────────


class DummyBehaviorDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset with IMU + Thermal + ToF batches."""

    def __init__(self, n_samples: int = 64, t_seq: int = 16, n_classes: int = 4) -> None:
        self.imu = torch.randn(n_samples, t_seq, 7)
        self.thermo = torch.randn(n_samples, t_seq, 5)
        self.tof = torch.randn(n_samples, t_seq, 320)
        self.tof_mask = (torch.randn(n_samples, t_seq, 320) > 0).float()
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "imu": self.imu[idx],
            "thermo": self.thermo[idx],
            "tof": self.tof[idx],
            "tof_mask": self.tof_mask[idx],
            "label": self.labels[idx],
        }


class SimpleTestModel(nn.Module):
    """Tiny model for trainer tests — not a real baseline."""

    def __init__(self, in_features: int = 7, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(
        self,
        imu: torch.Tensor,
        thermo: torch.Tensor | None = None,
        tof: torch.Tensor | None = None,
        tof_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Ignore thermo, tof — just use IMU mean
        return self.fc(imu.mean(dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_loaders():
    """Tiny train / val / test DataLoaders."""
    ds_train = DummyBehaviorDataset(n_samples=32, t_seq=16)
    ds_val = DummyBehaviorDataset(n_samples=16, t_seq=16)
    ds_test = DummyBehaviorDataset(n_samples=16, t_seq=16)

    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=8, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=8, shuffle=False)
    return train_loader, val_loader, test_loader


class TestTrainer:
    def test_train_epoch_returns_metrics(self, tiny_loaders):
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        train_loader, val_loader, _ = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=4,
            max_epochs=2,
            mixed_precision=False,
        )

        metrics = trainer.train_epoch(train_loader, epoch=0)
        assert "train/loss" in metrics
        assert "train/f1_macro" in metrics
        assert "train/accuracy" in metrics
        assert isinstance(metrics["train/loss"], float)

    def test_validate_epoch_returns_metrics(self, tiny_loaders):
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        _, val_loader, _ = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=4,
            max_epochs=2,
            mixed_precision=False,
        )

        metrics = trainer.validate_epoch(val_loader, epoch=0)
        assert "val/loss" in metrics
        assert "val/f1_macro" in metrics
        assert "val/f1_weighted" in metrics
        assert "val/accuracy" in metrics

    def test_test_epoch_returns_confusion_matrix(self, tiny_loaders):
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        _, _, test_loader = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)
        device = torch.device("cpu")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=4,
            max_epochs=1,
            mixed_precision=False,
        )

        result = trainer.test_epoch(test_loader)
        assert "confusion_matrix" in result
        assert result["confusion_matrix"].shape == (4, 4)

    def test_fit_2_epochs(self, tiny_loaders):
        from arst.training.callbacks import MetricTracker
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        train_loader, val_loader, _ = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)
        device = torch.device("cpu")
        tracker = MetricTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                num_classes=4,
                max_epochs=2,
                mixed_precision=False,
                callbacks=[tracker],
                checkpoint_dir=Path(tmpdir),
            )

            history = trainer.fit(train_loader, val_loader)

        assert len(tracker.get("val/f1_macro")) == 2

    def test_early_stopping_triggers(self, tiny_loaders):
        from arst.training.callbacks import EarlyStopping, MetricTracker
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        train_loader, val_loader, _ = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)
        device = torch.device("cpu")

        early_stopping = EarlyStopping(
            monitor="val/f1_macro",
            patience=1,
            mode="max",
        )
        tracker = MetricTracker()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                num_classes=4,
                max_epochs=20,
                mixed_precision=False,
                callbacks=[early_stopping, tracker],
                checkpoint_dir=Path(tmpdir),
            )
            trainer.fit(train_loader, val_loader)

        # With patience=1 on a tiny model, should stop well before 20 epochs
        assert len(tracker.get("val/f1_macro")) < 20

    def test_checkpoint_saved(self, tiny_loaders):
        from arst.training.callbacks import ModelCheckpoint
        from arst.training.losses import FocalLoss
        from arst.training.trainer import Trainer

        train_loader, val_loader, _ = tiny_loaders
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = FocalLoss(gamma=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            ckpt_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val/f1_macro",
                mode="max",
                save_top_k=1,
            )
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
                num_classes=4,
                max_epochs=3,
                mixed_precision=False,
                callbacks=[ckpt_callback],
                checkpoint_dir=ckpt_dir,
            )
            trainer.fit(train_loader, val_loader)

            # "last.pt" should always exist
            assert (ckpt_dir / "last.pt").exists()


class TestExperimentContext:
    def test_seed_sets_random_state(self):
        from arst.training.experiment import seed_everything

        seed_everything(42)
        val1 = torch.randn(3).tolist()
        seed_everything(42)
        val2 = torch.randn(3).tolist()
        assert val1 == val2

    def test_get_device_returns_valid(self):
        from arst.training.experiment import get_device

        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")

    def test_make_run_id_format(self):
        from arst.training.experiment import make_run_id

        run_id = make_run_id("mlp", seed=42)
        assert "mlp" in run_id
        assert "42" in run_id
