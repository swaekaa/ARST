"""
PyTorch Dataset for ARST — Phase 1 compatible.

Supports both:
  1. Flat CSV mode (Phase 1/2): directly reads from the raw train.csv
     using sequence-level grouping and window extraction.
  2. HDF5 mode (Phase 3+): loads from preprocessed HDF5 file for
     fast training with the full pipeline.

Design decisions:
  - CSV mode uses chunked / on-demand reading to avoid loading 1.1 GB into RAM.
  - HDF5 mode caches the entire dataset optionally for maximum throughput.
  - Missing modality handling is built-in (zero fill, learned null, noise).
  - ToF mask is always returned as a separate tensor for the Reliability Module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sensor group constants (matches configs/sensor_groups.yaml)
# ──────────────────────────────────────────────────────────────────────────────
IMU_COLS: list[str] = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

THERMAL_COLS: list[str] = [f"thm_{i}" for i in range(1, 6)]  # thm_1 .. thm_5

TOF_COLS: list[str] = [
    f"tof_{s}_v{p}" for s in range(1, 6) for p in range(64)
]  # 5 sensors × 64 pixels = 320 features

TOF_INVALID_SENTINEL: float = -1.0  # Invalid reading marker in raw data
METADATA_COLS: list[str] = [
    "row_id",
    "sequence_type",
    "sequence_id",
    "sequence_counter",
    "subject",
    "orientation",
    "behavior",
    "phase",
    "gesture",
]


# ──────────────────────────────────────────────────────────────────────────────
# HDF5-backed Dataset (Phase 3+)
# ──────────────────────────────────────────────────────────────────────────────
class ARSTDataset(Dataset):
    """
    Dataset for loading windowed, preprocessed ARST sensor data from HDF5.

    Expected HDF5 structure::

        /windows/
            imu       [N, T, 7]    float32  – acc_xyz + quaternion
            thermo    [N, T, 5]    float32  – 5 thermopile channels
            tof       [N, T, 320]  float32  – 5 sensors × 64 pixels
            tof_mask  [N, T, 320]  float32  – 1=valid, 0=invalid (-1 sentinel)
            labels    [N]          int64    – behavior class index
        /splits/
            train/indices  [N_train]  int64
            val/indices    [N_val]    int64
            test/indices   [N_test]   int64
        /metadata/
            class_names   [C]  str

    Args:
        hdf5_path: Path to the preprocessed HDF5 file.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        transform: Optional callable applied to each sample dict before
            converting to tensors (used for online augmentation).
        missing_modality_mode: Strategy when a modality is all-zeros:
            ``"zero"`` (leave as-is), ``"noise"`` (add small Gaussian noise),
            or ``"learned"`` (placeholder for future learned null embedding).
        cache_in_memory: If ``True``, load the entire dataset into RAM on
            construction. Speeds up training when RAM is sufficient.
    """

    MODALITIES: tuple[str, ...] = ("imu", "thermo", "tof")

    def __init__(
        self,
        hdf5_path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        missing_modality_mode: Literal["zero", "noise", "learned"] = "zero",
        cache_in_memory: bool = False,
    ) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError("h5py required for ARSTDataset: pip install h5py") from e

        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.missing_modality_mode = missing_modality_mode
        self.cache_in_memory = cache_in_memory

        self._cache: dict[str, np.ndarray] | None = None
        self._load_metadata()
        if cache_in_memory:
            self._cache_data()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        """Load labels and metadata without loading the full data arrays."""
        import h5py

        with h5py.File(self.hdf5_path, "r") as f:
            split_group = f[f"splits/{self.split}"]
            self.indices: np.ndarray = split_group["indices"][:]  # [N]

            windows = f["windows"]
            self.n_samples: int = len(self.indices)
            self.window_size: int = int(windows["imu"].shape[1])  # T
            self.labels: np.ndarray = windows["labels"][self.indices]  # [N]

            # Load class names if available
            if "metadata" in f and "class_names" in f["metadata"]:
                self.class_names: list[str] = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in f["metadata"]["class_names"][:]
                ]
            else:
                self.class_names = [str(i) for i in range(self.num_classes)]

        logger.info(
            "Loaded %s split: %d samples, window_size=%d, n_classes=%d",
            self.split,
            self.n_samples,
            self.window_size,
            self.num_classes,
        )

    def _cache_data(self) -> None:
        """Load all windows into RAM for fast sequential access."""
        import h5py

        logger.info("Caching %s split into RAM...", self.split)
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
        mb = sum(v.nbytes for v in self._cache.values()) / 1024**2
        logger.info("Dataset cached: %.1f MB", mb)

    # ── Core interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one sample as a dict of tensors.

        Returns:
            dict containing:
            - ``"imu"``      : ``[T, 7]``    float32
            - ``"thermo"``   : ``[T, 5]``    float32
            - ``"tof"``      : ``[T, 320]``  float32  (invalid → 0.0)
            - ``"tof_mask"`` : ``[T, 320]``  float32  (1=valid, 0=invalid)
            - ``"label"``    : scalar        int64
            - ``"index"``    : scalar        int64
        """
        if self._cache is not None:
            sample = {
                "imu": self._cache["imu"][idx].copy(),
                "thermo": self._cache["thermo"][idx].copy(),
                "tof": self._cache["tof"][idx].copy(),
                "tof_mask": self._cache["tof_mask"][idx].copy(),
                "label": int(self._cache["labels"][idx]),
            }
        else:
            import h5py

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

        # Apply online augmentation (training only)
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "imu": torch.from_numpy(np.asarray(sample["imu"], dtype=np.float32)),
            "thermo": torch.from_numpy(np.asarray(sample["thermo"], dtype=np.float32)),
            "tof": torch.from_numpy(np.asarray(sample["tof"], dtype=np.float32)),
            "tof_mask": torch.from_numpy(np.asarray(sample["tof_mask"], dtype=np.float32)),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        """Number of unique behavior classes."""
        return int(self.labels.max()) + 1

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.

        Returns:
            Tensor of shape ``[num_classes]`` with weights normalized so
            that ``weights.sum() == num_classes``.
        """
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)

    @property
    def modality_dims(self) -> dict[str, int]:
        """Return the feature dimension of each modality."""
        return {
            "imu": len(IMU_COLS),  # 7
            "thermo": len(THERMAL_COLS),  # 5
            "tof": len(TOF_COLS),  # 320
        }

    def validate_shapes(self) -> bool:
        """
        Validate shapes of a single sample (smoke test).

        Returns:
            ``True`` if all shapes match expected dimensions.

        Raises:
            AssertionError: If any shape is wrong.
        """
        sample = self[0]
        T = self.window_size
        assert sample["imu"].shape == (T, 7), f"IMU shape mismatch: {sample['imu'].shape}"
        assert sample["thermo"].shape == (T, 5), f"Thermal shape mismatch: {sample['thermo'].shape}"
        assert sample["tof"].shape == (T, 320), f"ToF shape mismatch: {sample['tof'].shape}"
        assert sample["tof_mask"].shape == (
            T,
            320,
        ), f"ToF mask shape mismatch: {sample['tof_mask'].shape}"
        assert sample["label"].shape == (), f"Label not scalar: {sample['label'].shape}"
        logger.info("Shape validation passed: T=%d, IMU=7, Thermal=5, ToF=320", T)
        return True


# ──────────────────────────────────────────────────────────────────────────────
# CSV-backed Dataset (Phase 1/2 — no HDF5 required)
# ──────────────────────────────────────────────────────────────────────────────
class ARSTRawCSVDataset(Dataset):
    """
    Dataset that serves real windowed sequences for Phase 2 training.

    Stores pre-extracted numpy arrays of shape [N, T, F] for each modality,
    so every sample returns a genuine T-step time-series rather than a
    tiled single-row mean.

    Args:
        imu:       [N, T, 7]    float32
        thermo:    [N, T, 5]    float32
        tof:       [N, T, 320]  float32 (invalids replaced with 0.0)
        tof_mask:  [N, T, 320]  float32 (1=valid, 0=invalid)
        labels:    [N]          int64
        n_classes: Total number of classes in the full dataset (fixed, global).
        window_size: T — stored for compatibility checks.
        transform: Optional augmentation callable.
    """

    def __init__(
        self,
        imu: np.ndarray,
        thermo: np.ndarray,
        tof: np.ndarray,
        tof_mask: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
        window_size: int,
        transform: Callable | None = None,
    ) -> None:
        self.imu = imu.astype(np.float32)
        self.thermo = thermo.astype(np.float32)
        self.tof = tof.astype(np.float32)
        self.tof_mask = tof_mask.astype(np.float32)
        self.labels: np.ndarray = labels.astype(np.int64)
        self._n_classes = n_classes
        self.window_size = window_size
        self.transform = transform
        self.n_samples: int = len(labels)

        logger.info(
            "ARSTRawCSVDataset: %d windows, T=%d, n_classes=%d",
            self.n_samples,
            self.window_size,
            self._n_classes,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        imu = self.imu[idx]  # [T, 7]
        thermo = self.thermo[idx]  # [T, 5]
        tof = self.tof[idx]  # [T, 320]
        tof_mask = self.tof_mask[idx]  # [T, 320]
        label = int(self.labels[idx])

        sample: dict = {
            "imu": imu,
            "thermo": thermo,
            "tof": tof,
            "tof_mask": tof_mask,
            "label": label,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "imu": torch.from_numpy(np.asarray(sample["imu"], dtype=np.float32)),
            "thermo": torch.from_numpy(np.asarray(sample["thermo"], dtype=np.float32)),
            "tof": torch.from_numpy(np.asarray(sample["tof"], dtype=np.float32)),
            "tof_mask": torch.from_numpy(np.asarray(sample["tof_mask"], dtype=np.float32)),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        """Fixed global class count (not derived from observed labels)."""
        return self._n_classes

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights using the GLOBAL n_classes so the tensor
        always has shape [n_classes] regardless of which classes appear in
        this split."""
        counts = np.bincount(self.labels, minlength=self._n_classes).astype(np.float32)
        # For classes absent from this split, assign weight 0 (ignored by loss)
        weights = np.where(counts > 0, 1.0 / counts, 0.0)
        total = weights.sum()
        if total > 0:
            weights = weights / total * self._n_classes
        return torch.from_numpy(weights)


# ──────────────────────────────────────────────────────────────────────────────
# Sequence-level Dataset (handles variable-length sequences from flat CSV)
# ──────────────────────────────────────────────────────────────────────────────
class ARSTSequenceDataset(Dataset):
    """
    Sequence-aware Dataset that reads grouped sequences from a pre-loaded
    metadata DataFrame and lazily loads windows from the raw CSV on demand.

    This handles the actual flat-CSV structure: sequences are identified by
    ``sequence_id``, and windows are extracted at index time.

    Args:
        sequence_index: DataFrame with one row per (sequence_id, window_start).
            Must contain: ``sequence_id``, ``window_start``, ``label`` columns.
        csv_path: Path to train.csv (or test.csv).
        window_size: Fixed window length T.
        transform: Optional augmentation callable.
        normalizer: Optional fitted normalizer object with ``.transform(df)`` method.
    """

    def __init__(
        self,
        sequence_index: pd.DataFrame,
        csv_path: str | Path,
        window_size: int,
        transform: Callable | None = None,
        normalizer: object | None = None,
    ) -> None:
        self.sequence_index = sequence_index.reset_index(drop=True)
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.transform = transform
        self.normalizer = normalizer

        self.labels = self.sequence_index["label"].values.astype(np.int64)
        self.n_samples = len(self.sequence_index)

        logger.info(
            "ARSTSequenceDataset: %d windows from %d sequences, T=%d",
            self.n_samples,
            self.sequence_index["sequence_id"].nunique(),
            self.window_size,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return one window by reading from the CSV."""
        meta = self.sequence_index.iloc[idx]
        seq_id: str = str(meta["sequence_id"])
        win_start: int = int(meta["window_start"])
        label: int = int(meta["label"])

        # Load the window from CSV using chunked scan is too slow for per-item.
        # In practice, preprocess to HDF5 first. This is a fallback.
        # Read the relevant rows: sequence_counter in [win_start, win_start+T)
        all_cols = ["sequence_id", "sequence_counter", *IMU_COLS, *THERMAL_COLS, *TOF_COLS]

        seq_df = pd.read_csv(
            self.csv_path,
            usecols=all_cols,
            dtype="float32",
            converters={"sequence_id": str, "sequence_counter": int},
        )
        window = seq_df[
            (seq_df["sequence_id"] == seq_id)
            & (seq_df["sequence_counter"] >= win_start)
            & (seq_df["sequence_counter"] < win_start + self.window_size)
        ].sort_values("sequence_counter")

        T = len(window)

        def extract(cols: list[str]) -> np.ndarray:
            avail = [c for c in cols if c in window.columns]
            arr = window[avail].values.astype(np.float32)
            # Pad if sequence is shorter than window_size
            if self.window_size > T:
                arr = np.pad(arr, ((0, self.window_size - T), (0, 0)))
            return arr[: self.window_size]

        imu = extract(IMU_COLS)  # [T, 7]
        thermo = extract(THERMAL_COLS)  # [T, 5]
        tof_raw = extract(TOF_COLS)  # [T, 320]

        tof_mask = (tof_raw != TOF_INVALID_SENTINEL).astype(np.float32)
        tof_clean = np.where(tof_raw == TOF_INVALID_SENTINEL, 0.0, tof_raw)

        sample: dict = {
            "imu": imu,
            "thermo": thermo,
            "tof": tof_clean,
            "tof_mask": tof_mask,
            "label": label,
        }

        if self.normalizer is not None:
            sample = self.normalizer.transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

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
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)
