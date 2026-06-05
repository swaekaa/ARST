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

from arst.data.dataset import IMU_COLS, THERMAL_COLS, TOF_COLS, ARSTDataset, ARSTRawCSVDataset

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

    def make_window_df(subjects_set: set[str]) -> pd.DataFrame:
        """Extract one-row-per-window DataFrame for the given subjects."""
        sub_df = df[df["subject"].isin(subjects_set)].copy()
        windows = []
        for seq_id, seq_data in sub_df.groupby("sequence_id"):
            seq_data = seq_data.sort_values("sequence_counter")
            label = int(seq_data["label"].mode()[0])
            # Aggregate: use up to window_size rows, compute column mean ignoring NaN.
            # Using values[0] was the NaN root cause: 6.8% of sequences have NaN
            # in their first row, which propagates directly into training tensors.
            window_data = seq_data.iloc[:window_size]
            row: dict = {"sequence_id": seq_id, "label": label}
            for col in IMU_COLS + THERMAL_COLS + TOF_COLS:
                if col in window_data.columns:
                    col_vals = window_data[col].values
                    if len(col_vals) == 0 or np.all(np.isnan(col_vals)):
                        row[col] = 0.0
                    else:
                        mean_val = float(np.nanmean(col_vals))
                        row[col] = mean_val if np.isfinite(mean_val) else 0.0
                else:
                    row[col] = 0.0
            windows.append(row)
        return pd.DataFrame(windows)

    train_df = make_window_df(train_subjects)
    val_df = make_window_df(val_subjects)
    test_df = make_window_df(test_subjects)

    logger.info(
        "  Splits: train=%d, val=%d, test=%d windows",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_ds = ARSTRawCSVDataset(train_df, window_size=window_size)
    val_ds = ARSTRawCSVDataset(val_df, window_size=window_size)
    test_ds = ARSTRawCSVDataset(test_df, window_size=window_size)

    n_classes = len(behavior_encoder)
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
