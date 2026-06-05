"""
DataLoader module for ARST — Phase 1 compatible.

Provides:
  - :class:`ARSTDataModule`: HDF5-backed train/val/test loaders (Phase 3+).
  - :func:`build_csv_loaders`: Quick CSV-backed loaders for Phase 1/2 EDA.

Design decisions:
  - Separates data module concerns from model training.
  - Supports both HDF5 (fast, preprocessed) and raw CSV (flexible, slow) modes.
  - Class weights are exposed for weighted loss functions.
  - Compatible with PyTorch Lightning-style setup() / teardown() convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from arst.data.dataset import (
    IMU_COLS,
    THERMAL_COLS,
    TOF_COLS,
    TOF_INVALID_SENTINEL,
    ARSTDataset,
    ARSTRawCSVDataset,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HDF5-backed DataModule (primary, Phase 3+)
# ──────────────────────────────────────────────────────────────────────────────
class ARSTDataModule:
    """
    Manages :class:`ARSTDataset` instances and :class:`DataLoader` objects
    for all three splits (train / val / test).

    Args:
        hdf5_path: Path to the preprocessed HDF5 file.
        batch_size: Training batch size.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Pin GPU-accessible memory in DataLoader.
        persistent_workers: Keep worker processes alive between epochs.
        augmentation_config: Config dict forwarded to the augmentation pipeline.
        cache_in_memory: Pre-load entire dataset into RAM (fast if RAM allows).
        use_weighted_sampler: Balance training batches by class frequency.
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
        use_weighted_sampler: bool = False,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.augmentation_config = augmentation_config or {}
        self.cache_in_memory = cache_in_memory
        self.use_weighted_sampler = use_weighted_sampler

        self._train_dataset: ARSTDataset | None = None
        self._val_dataset: ARSTDataset | None = None
        self._test_dataset: ARSTDataset | None = None

    def setup(self) -> None:
        """Initialize all three dataset splits."""
        try:
            from arst.data.augmentation import build_augmentation_pipeline

            train_transform = build_augmentation_pipeline(
                self.augmentation_config, is_training=True
            )
            eval_transform = build_augmentation_pipeline(
                self.augmentation_config, is_training=False
            )
        except ImportError:
            logger.warning("augmentation module not found — running without transforms")
            train_transform = None
            eval_transform = None

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
            "DataModule ready: train=%d, val=%d, test=%d samples | classes=%d",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
            self.num_classes,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with optional class-balanced sampling."""
        assert self._train_dataset is not None, "Call setup() first"
        sampler = None
        shuffle = True

        if self.use_weighted_sampler:
            weights = self._train_dataset.class_weights
            sample_weights = weights[torch.from_numpy(self._train_dataset.labels)]
            sampler = WeightedRandomSampler(
                weights=sample_weights.double(),
                num_samples=len(self._train_dataset),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler
            logger.info("Using WeightedRandomSampler for class-balanced training")

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (no shuffle, 2× batch size)."""
        assert self._val_dataset is not None, "Call setup() first"
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader (no shuffle, 2× batch size)."""
        assert self._test_dataset is not None, "Call setup() first"
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        """Number of unique behavior classes."""
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.num_classes

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss functions."""
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.class_weights

    @property
    def modality_dims(self) -> dict[str, int]:
        """Feature dimension of each modality."""
        return {"imu": 7, "thermo": 5, "tof": 320}

    @property
    def window_size(self) -> int:
        """Fixed window length T."""
        assert self._train_dataset is not None, "Call setup() first"
        return self._train_dataset.window_size


# ──────────────────────────────────────────────────────────────────────────────
# CSV-backed loaders (Phase 1/2 — no HDF5 required)
# ──────────────────────────────────────────────────────────────────────────────
def build_csv_loaders(
    csv_path: str | Path,
    window_size: int = 128,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    behavior_encoder: dict[str, int] | None = None,
    max_rows: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train/val/test DataLoaders directly from the raw CSV.

    .. warning::
        This is intended for Phase 1/2 EDA and quick experiments only.
        The CSV is loaded into RAM (up to ``max_rows`` rows).
        For full training, convert to HDF5 and use :class:`ARSTDataModule`.

    Args:
        csv_path: Path to train.csv.
        window_size: Window length T for padding/truncation.
        batch_size: Training batch size.
        val_fraction: Fraction of sequences for validation.
        test_fraction: Fraction of sequences for test.
        num_workers: DataLoader worker processes (0 = main process).
        seed: Random seed for reproducibility.
        behavior_encoder: Optional mapping from behavior string to int.
            If ``None``, computed automatically from training data.
        max_rows: Limit rows read (for quick debugging). ``None`` = all.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict) where
        ``info_dict`` contains ``n_classes``, ``class_weights``,
        ``behavior_encoder``.
    """
    logger.info("Building CSV-backed loaders from %s (max_rows=%s)", csv_path, max_rows)

    # Load data (subset for EDA, all data for training)
    usecols = (
        ["sequence_id", "sequence_counter", "subject", "behavior"]
        + IMU_COLS
        + THERMAL_COLS
        + TOF_COLS
    )
    # Filter to only columns that exist
    header_df = pd.read_csv(csv_path, nrows=1)
    usecols = [c for c in usecols if c in header_df.columns]

    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    # Encode behavior labels
    if behavior_encoder is None:
        behaviors = sorted(df["behavior"].dropna().unique())
        behavior_encoder = {b: i for i, b in enumerate(behaviors)}
    df["label"] = df["behavior"].map(behavior_encoder).fillna(-1).astype(int)
    df = df[df["label"] >= 0]  # drop unlabeled

    # Subject-based split (prevents data leakage)
    subjects = df["subject"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    n_classes = len(behavior_encoder)  # global, fixed regardless of split content
    imu_cols_avail = [c for c in IMU_COLS if c in df.columns]
    thm_cols_avail = [c for c in THERMAL_COLS if c in df.columns]
    tof_cols_avail = [c for c in TOF_COLS if c in df.columns]

    def extract_split_arrays(
        subjects_set: set,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each sequence in the split, extract a [T, F] array of real sensor
        readings. Returns five stacked arrays: imu, thermo, tof, tof_mask, labels.

        - Sequences shorter than window_size are zero-padded on the right.
        - Sequences longer than window_size are truncated to the first T rows.
        - NaN values are filled with 0.0 before returning.
        """
        sub_df = df[df["subject"].isin(subjects_set)]
        all_imu, all_thm, all_tof, all_mask, all_labels = [], [], [], [], []

        for _seq_id, seq_data in sub_df.groupby("sequence_id"):
            seq_data = seq_data.sort_values("sequence_counter")
            label = int(seq_data["label"].mode()[0])

            # Truncate or keep up to window_size rows
            window = seq_data.iloc[:window_size]
            T_actual = len(window)

            # --- IMU [T, 7] ---
            if imu_cols_avail:
                imu_raw = window[imu_cols_avail].values.astype(np.float32)  # [T_actual, n_imu]
            else:
                imu_raw = np.zeros((T_actual, len(IMU_COLS)), dtype=np.float32)
            # Pad missing cols to exactly len(IMU_COLS)
            if imu_raw.shape[1] < len(IMU_COLS):
                imu_raw = np.pad(imu_raw, ((0, 0), (0, len(IMU_COLS) - imu_raw.shape[1])))
            # Pad time dimension
            if T_actual < window_size:
                imu_raw = np.pad(imu_raw, ((0, window_size - T_actual), (0, 0)))
            np.nan_to_num(imu_raw, copy=False, nan=0.0)

            # --- Thermal [T, 5] ---
            if thm_cols_avail:
                thm_raw = window[thm_cols_avail].values.astype(np.float32)
            else:
                thm_raw = np.zeros((T_actual, len(THERMAL_COLS)), dtype=np.float32)
            if thm_raw.shape[1] < len(THERMAL_COLS):
                thm_raw = np.pad(thm_raw, ((0, 0), (0, len(THERMAL_COLS) - thm_raw.shape[1])))
            if T_actual < window_size:
                thm_raw = np.pad(thm_raw, ((0, window_size - T_actual), (0, 0)))
            np.nan_to_num(thm_raw, copy=False, nan=0.0)

            # --- ToF [T, 320] with invalidity mask ---
            if tof_cols_avail:
                tof_raw = window[tof_cols_avail].values.astype(np.float32)  # [T_actual, n_tof]
            else:
                tof_raw = np.zeros((T_actual, len(TOF_COLS)), dtype=np.float32)
            if tof_raw.shape[1] < len(TOF_COLS):
                tof_raw = np.pad(tof_raw, ((0, 0), (0, len(TOF_COLS) - tof_raw.shape[1])))
            if T_actual < window_size:
                tof_raw = np.pad(tof_raw, ((0, window_size - T_actual), (0, 0)))
            # Mask: sentinel -1.0 = invalid
            tof_mask = (tof_raw != TOF_INVALID_SENTINEL).astype(np.float32)
            tof_clean = np.where(tof_raw == TOF_INVALID_SENTINEL, 0.0, tof_raw)
            np.nan_to_num(tof_clean, copy=False, nan=0.0)
            np.nan_to_num(tof_mask, copy=False, nan=0.0)

            all_imu.append(imu_raw)
            all_thm.append(thm_raw)
            all_tof.append(tof_clean)
            all_mask.append(tof_mask)
            all_labels.append(label)

        if not all_labels:
            # Empty split (can happen with very small max_rows)
            return (
                np.zeros((0, window_size, len(IMU_COLS)), np.float32),
                np.zeros((0, window_size, len(THERMAL_COLS)), np.float32),
                np.zeros((0, window_size, len(TOF_COLS)), np.float32),
                np.zeros((0, window_size, len(TOF_COLS)), np.float32),
                np.zeros((0,), np.int64),
            )

        return (
            np.stack(all_imu),  # [N, T, 7]
            np.stack(all_thm),  # [N, T, 5]
            np.stack(all_tof),  # [N, T, 320]
            np.stack(all_mask),  # [N, T, 320]
            np.array(all_labels, dtype=np.int64),
        )

    logger.info("  Extracting sequence windows for each split...")
    tr_imu, tr_thm, tr_tof, tr_mask, tr_lbl = extract_split_arrays(train_subjects)
    va_imu, va_thm, va_tof, va_mask, va_lbl = extract_split_arrays(val_subjects)
    te_imu, te_thm, te_tof, te_mask, te_lbl = extract_split_arrays(test_subjects)

    logger.info(
        "  Splits: train=%d, val=%d, test=%d windows",
        len(tr_lbl),
        len(va_lbl),
        len(te_lbl),
    )

    train_ds = ARSTRawCSVDataset(
        imu=tr_imu,
        thermo=tr_thm,
        tof=tr_tof,
        tof_mask=tr_mask,
        labels=tr_lbl,
        n_classes=n_classes,
        window_size=window_size,
    )
    val_ds = ARSTRawCSVDataset(
        imu=va_imu,
        thermo=va_thm,
        tof=va_tof,
        tof_mask=va_mask,
        labels=va_lbl,
        n_classes=n_classes,
        window_size=window_size,
    )
    test_ds = ARSTRawCSVDataset(
        imu=te_imu,
        thermo=te_thm,
        tof=te_tof,
        tof_mask=te_mask,
        labels=te_lbl,
        n_classes=n_classes,
        window_size=window_size,
    )

    # Class weights from TRAIN split only, always [n_classes] shape
    class_weights = train_ds.class_weights

    _loader_kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        **_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        **_loader_kwargs,
    )

    info = {
        "n_classes": n_classes,
        "class_weights": class_weights,
        "behavior_encoder": behavior_encoder,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "modality_dims": {"imu": 7, "thermo": 5, "tof": 320},
        "window_size": window_size,
    }
    logger.info(
        "  n_classes=%d, class_weights range=[%.3f, %.3f]",
        n_classes,
        class_weights.min().item(),
        class_weights.max().item(),
    )
    return train_loader, val_loader, test_loader, info
