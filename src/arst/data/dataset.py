"""
PyTorch Dataset for ARST.

Handles:
  - Loading preprocessed HDF5 windows
  - Online augmentation (optional)
  - Missing modality masking
  - Returns consistent tensors for all three modalities
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ARSTDataset(Dataset):
    """
    Dataset for loading windowed, preprocessed ARST sensor data from HDF5.

    Expected HDF5 structure:
        /windows/
            imu      [N, T, 6]    float32
            thermo   [N, T, 64]   float32
            tof      [N, T, 64]   float32
            tof_mask [N, T, 64]   float32   (1=valid, 0=invalid)
            labels   [N]          int64
        /metadata/
            sequence_ids  [N]     str
            subject_ids   [N]     str
            window_starts [N]     int64

    Args:
        hdf5_path: Path to the preprocessed HDF5 file.
        split: "train", "val", or "test".
        transform: Optional callable for online augmentation.
        missing_modality_mode: "zero" | "noise" | "learned".
            Applied when a modality is all-zeros (from preprocessing placeholder).
        cache_in_memory: Load entire dataset into RAM for faster I/O.
    """

    MODALITIES = ("imu", "thermo", "tof")

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        missing_modality_mode: str = "zero",
        cache_in_memory: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.missing_modality_mode = missing_modality_mode
        self.cache_in_memory = cache_in_memory

        # Load metadata and optionally cache data
        self._load_metadata()
        if cache_in_memory:
            self._cache_data()

    def _load_metadata(self) -> None:
        """Load labels and metadata without loading the full data arrays."""
        with h5py.File(self.hdf5_path, "r") as f:
            split_group = f[f"splits/{self.split}"]
            self.indices = split_group["indices"][:]  # [N] → indices into /windows/

            windows = f["windows"]
            self.n_samples = len(self.indices)
            self.window_size = windows["imu"].shape[1]  # T
            self.labels = windows["labels"][self.indices]  # [N]

        logger.info(
            "Loaded %s split: %d samples, window_size=%d",
            self.split,
            self.n_samples,
            self.window_size,
        )

    def _cache_data(self) -> None:
        """Load all data into memory for fast access."""
        logger.info("Caching dataset into memory...")
        with h5py.File(self.hdf5_path, "r") as f:
            windows = f["windows"]
            idx = self.indices
            self._cache = {
                "imu": windows["imu"][idx].astype(np.float32),
                "thermo": windows["thermo"][idx].astype(np.float32),
                "tof": windows["tof"][idx].astype(np.float32),
                "tof_mask": windows["tof_mask"][idx].astype(np.float32),
                "labels": windows["labels"][idx].astype(np.int64),
            }
        logger.info("Dataset cached (%.1f MB).", self._estimate_cache_mb())

    def _estimate_cache_mb(self) -> float:
        total_bytes = sum(v.nbytes for v in self._cache.values())
        return total_bytes / (1024**2)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                "imu"      : [T, 6]   float32
                "thermo"   : [T, 64]  float32
                "tof"      : [T, 64]  float32
                "tof_mask" : [T, 64]  float32
                "label"    : scalar int64
                "index"    : scalar int64 (global dataset index)
        """
        if self.cache_in_memory:
            sample = {
                "imu": self._cache["imu"][idx].copy(),
                "thermo": self._cache["thermo"][idx].copy(),
                "tof": self._cache["tof"][idx].copy(),
                "tof_mask": self._cache["tof_mask"][idx].copy(),
                "label": int(self._cache["labels"][idx]),
            }
        else:
            global_idx = int(self.indices[idx])
            with h5py.File(self.hdf5_path, "r") as f:
                windows = f["windows"]
                sample = {
                    "imu": windows["imu"][global_idx].astype(np.float32),
                    "thermo": windows["thermo"][global_idx].astype(np.float32),
                    "tof": windows["tof"][global_idx].astype(np.float32),
                    "tof_mask": windows["tof_mask"][global_idx].astype(np.float32),
                    "label": int(windows["labels"][global_idx]),
                }

        # Apply augmentation (training only)
        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to tensors
        return {
            "imu": torch.from_numpy(sample["imu"]),
            "thermo": torch.from_numpy(sample["thermo"]),
            "tof": torch.from_numpy(sample["tof"]),
            "tof_mask": torch.from_numpy(sample["tof_mask"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)
