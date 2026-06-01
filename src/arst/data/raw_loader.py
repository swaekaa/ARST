"""
Raw data loader for CMI sensor data.

Handles loading of parquet files, timestamp alignment across modalities,
and missing file/modality detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModalityData:
    """Container for one modality's raw time-series data."""

    def __init__(self, data: np.ndarray, timestamps: np.ndarray, channels: list[str]):
        self.data = data  # [T, C]
        self.timestamps = timestamps  # [T]
        self.channels = channels
        self.is_missing = False

    @classmethod
    def missing(cls, n_channels: int, channel_names: list[str]) -> ModalityData:
        """Create a zero-filled placeholder for a missing modality."""
        obj = cls(
            data=np.zeros((0, n_channels), dtype=np.float32),
            timestamps=np.array([], dtype=np.float64),
            channels=channel_names,
        )
        obj.is_missing = True
        return obj


class RawSequenceLoader:
    """
    Load raw sensor data for a single sequence from disk.

    Expects the following directory structure:
        <root_dir>/<sequence_id>/accel.parquet
        <root_dir>/<sequence_id>/gyro.parquet
        <root_dir>/<sequence_id>/thermo.parquet
        <root_dir>/<sequence_id>/tof.parquet

    Args:
        root_dir: Path to the split directory (e.g., data/raw/sensor_data/train/)
        imu_channels: Column names for accelerometer and gyroscope.
        thermal_channels: Column names for thermopile (64 pixel values).
        tof_channels: Column names for ToF sensor (64 pixel values).
        timestamp_col: Name of the timestamp column in each parquet file.
    """

    IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    THERMAL_CHANNELS = [f"pixel_{i}" for i in range(64)]
    TOF_CHANNELS = [f"tof_{i}" for i in range(64)]

    def __init__(
        self,
        root_dir: str | Path,
        timestamp_col: str = "timestamp",
    ):
        self.root_dir = Path(root_dir)
        self.timestamp_col = timestamp_col

    def load(self, sequence_id: str) -> dict[str, ModalityData]:
        """
        Load all modalities for a sequence.

        Returns:
            dict with keys "imu", "thermopile", "tof", each a ModalityData.
        """
        seq_dir = self.root_dir / sequence_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

        return {
            "imu": self._load_imu(seq_dir),
            "thermopile": self._load_thermopile(seq_dir),
            "tof": self._load_tof(seq_dir),
        }

    def _load_imu(self, seq_dir: Path) -> ModalityData:
        """Load and merge accelerometer + gyroscope parquet files."""
        accel_path = seq_dir / "accel.parquet"
        gyro_path = seq_dir / "gyro.parquet"

        if not accel_path.exists() or not gyro_path.exists():
            logger.warning("IMU files missing in %s — using zero placeholder.", seq_dir)
            return ModalityData.missing(6, self.IMU_CHANNELS)

        accel = pd.read_parquet(accel_path)
        gyro = pd.read_parquet(gyro_path)

        # Merge on timestamp
        merged = pd.merge_asof(
            accel.sort_values(self.timestamp_col),
            gyro.sort_values(self.timestamp_col),
            on=self.timestamp_col,
            direction="nearest",
            tolerance=pd.Timedelta("20ms"),
        )

        timestamps = merged[self.timestamp_col].values.astype(np.float64)
        data = merged[self.IMU_CHANNELS].values.astype(np.float32)
        return ModalityData(data, timestamps, self.IMU_CHANNELS)

    def _load_thermopile(self, seq_dir: Path) -> ModalityData:
        """Load thermopile array parquet file."""
        path = seq_dir / "thermo.parquet"
        if not path.exists():
            logger.warning("Thermopile file missing in %s.", seq_dir)
            return ModalityData.missing(64, self.THERMAL_CHANNELS)

        df = pd.read_parquet(path)
        timestamps = df[self.timestamp_col].values.astype(np.float64)
        data = df[self.THERMAL_CHANNELS].values.astype(np.float32)
        return ModalityData(data, timestamps, self.THERMAL_CHANNELS)

    def _load_tof(self, seq_dir: Path) -> ModalityData:
        """Load time-of-flight array parquet file."""
        path = seq_dir / "tof.parquet"
        if not path.exists():
            logger.warning("ToF file missing in %s.", seq_dir)
            return ModalityData.missing(64, self.TOF_CHANNELS)

        df = pd.read_parquet(path)
        timestamps = df[self.timestamp_col].values.astype(np.float64)
        data = df[self.TOF_CHANNELS].values.astype(np.float32)
        return ModalityData(data, timestamps, self.TOF_CHANNELS)

    def list_sequences(self) -> list[str]:
        """Return all sequence IDs in the root directory."""
        return sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
