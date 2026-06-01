"""
DataModule for ARST: manages train/val/test DataLoaders.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from arst.data.augmentation import build_augmentation_pipeline
from arst.data.dataset import ARSTDataset
from arst.utils.logging import get_logger

logger = get_logger(__name__)


class ARSTDataModule:
    """
    Manages ARSTDataset instances and DataLoaders for all splits.

    Args:
        hdf5_path: Path to the processed HDF5 dataset.
        batch_size: Training batch size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin GPU memory in DataLoader.
        augmentation_config: Config dict for training augmentations.
        cache_in_memory: Load full dataset to RAM.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        augmentation_config: dict | None = None,
        cache_in_memory: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.augmentation_config = augmentation_config or {}
        self.cache_in_memory = cache_in_memory

        self._train_dataset: ARSTDataset | None = None
        self._val_dataset: ARSTDataset | None = None
        self._test_dataset: ARSTDataset | None = None

    def setup(self) -> None:
        """Initialize all dataset splits."""
        train_transform = build_augmentation_pipeline(self.augmentation_config, is_training=True)
        eval_transform = build_augmentation_pipeline(self.augmentation_config, is_training=False)

        self._train_dataset = ARSTDataset(
            hdf5_path=self.hdf5_path,
            split="train",
            transform=train_transform,
            cache_in_memory=self.cache_in_memory,
        )
        self._val_dataset = ARSTDataset(
            hdf5_path=self.hdf5_path,
            split="val",
            transform=eval_transform,
            cache_in_memory=self.cache_in_memory,
        )
        self._test_dataset = ARSTDataset(
            hdf5_path=self.hdf5_path,
            split="test",
            transform=eval_transform,
            cache_in_memory=self.cache_in_memory,
        )

        logger.info(
            "DataModule ready: train=%d, val=%d, test=%d samples",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def num_classes(self) -> int:
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.num_classes

    @property
    def class_weights(self) -> torch.Tensor:
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.class_weights
