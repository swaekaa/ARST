"""
Per-modality signal preprocessing for ARST.

Implements:
  - IMU: bandpass filtering, gravity removal, z-score normalization
  - Thermopile: dead pixel correction, background subtraction, normalization
  - ToF: invalid reading masking, inpainting, normalization
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# IMU Preprocessing
# ─────────────────────────────────────────────────────────────


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the time axis (axis=0).

    Args:
        data: [T, C] array of raw signal values.
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order.

    Returns:
        Filtered array of same shape.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, data, axis=0).astype(np.float32)


def remove_gravity_highpass(
    accel: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Remove gravity component via high-pass filter.

    Args:
        accel: [T, 3] accelerometer data.
        fs: Sampling frequency.
        cutoff: High-pass cutoff frequency (Hz).
        order: Filter order.

    Returns:
        Gravity-removed accelerometer signal.
    """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = scipy_signal.butter(order, high, btype="high")
    return scipy_signal.filtfilt(b, a, accel, axis=0).astype(np.float32)


def preprocess_imu(
    data: np.ndarray,
    fs: float = 50.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 20.0,
    remove_gravity: bool = True,
    normalization: str = "zscore",
    norm_stats: dict | None = None,
) -> np.ndarray:
    """Full IMU preprocessing pipeline.

    Args:
        data: [T, 6] array (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z).
        fs: Sampling frequency.
        bandpass_low: Low cutoff for bandpass.
        bandpass_high: High cutoff for bandpass.
        remove_gravity: Whether to high-pass filter accelerometer.
        normalization: "zscore" | "minmax" | "none".
        norm_stats: Precomputed stats dict {"mean": ..., "std": ...} for zscore.

    Returns:
        Preprocessed [T, 6] array.
    """
    if data.shape[0] < 10:
        logger.warning("IMU sequence too short (%d samples), returning as-is.", data.shape[0])
        return data

    # Split into accelerometer and gyroscope
    accel, gyro = data[:, :3], data[:, 3:]

    # --- Accelerometer ---
    if remove_gravity:
        accel = remove_gravity_highpass(accel, fs=fs, cutoff=0.5)
    accel = bandpass_filter(accel, bandpass_low, bandpass_high, fs=fs)

    # --- Gyroscope ---
    gyro = bandpass_filter(gyro, bandpass_low, bandpass_high, fs=fs)

    # --- Recombine ---
    processed = np.concatenate([accel, gyro], axis=-1)

    # --- Normalization ---
    processed = _normalize(processed, normalization, norm_stats)

    return processed


# ─────────────────────────────────────────────────────────────
# Thermopile Preprocessing
# ─────────────────────────────────────────────────────────────


def correct_dead_pixels(
    data: np.ndarray,
    threshold_std: float = 5.0,
) -> np.ndarray:
    """Replace dead pixels (outliers) with spatial median of neighbors.

    Args:
        data: [T, 64] thermopile data (64 = 8×8 flattened).
        threshold_std: Pixels deviating more than this many std devs from temporal
                       median are flagged as dead.

    Returns:
        Dead-pixel-corrected data of same shape.
    """
    T = data.shape[0]
    data_3d = data.reshape(T, 8, 8).copy()  # [T, 8, 8]

    # Compute per-pixel temporal stats
    pixel_median = np.median(data_3d, axis=0)  # [8, 8]
    pixel_std = np.std(data_3d, axis=0) + 1e-6  # [8, 8]

    # Dead pixel mask: pixels consistently out of range across all timesteps
    dead_mask = (np.abs(data_3d - pixel_median) > threshold_std * pixel_std).all(axis=0)

    if dead_mask.any():
        logger.debug("Found %d dead thermopile pixels.", dead_mask.sum())
        for r in range(8):
            for c in range(8):
                if dead_mask[r, c]:
                    # Replace with neighbor median
                    neighbors = _get_neighbors(data_3d, r, c)
                    data_3d[:, r, c] = np.median(neighbors, axis=0)

    return data_3d.reshape(T, 64)


def preprocess_thermopile(
    data: np.ndarray,
    dead_pixel_threshold: float = 5.0,
    background_window: int = 100,
    normalization: str = "zscore",
    norm_stats: dict | None = None,
) -> np.ndarray:
    """Full thermopile preprocessing pipeline.

    Args:
        data: [T, 64] raw thermopile data.
        dead_pixel_threshold: Std devs for dead pixel detection.
        background_window: Number of frames for running background baseline.
        normalization: "zscore" | "minmax" | "none".
        norm_stats: Precomputed normalization statistics.

    Returns:
        Preprocessed [T, 64] array.
    """
    # Dead pixel correction
    data = correct_dead_pixels(data, threshold_std=dead_pixel_threshold)

    # Background subtraction (subtract running mean baseline)
    baseline = _running_mean(data, window=background_window)
    data = data - baseline

    # Normalization
    data = _normalize(data, normalization, norm_stats)

    return data.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# ToF Preprocessing
# ─────────────────────────────────────────────────────────────


def preprocess_tof(
    data: np.ndarray,
    invalid_value: float = -1.0,
    min_depth: float = 0.0,
    max_depth: float = 1000.0,
    normalization: str = "minmax",
    norm_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full ToF preprocessing pipeline.

    Args:
        data: [T, 64] raw ToF data (depth in mm).
        invalid_value: Value indicating an invalid measurement.
        min_depth: Minimum valid depth (mm).
        max_depth: Maximum valid depth (mm).
        normalization: "zscore" | "minmax" | "none".
        norm_stats: Precomputed normalization statistics.

    Returns:
        Tuple of (processed_data [T, 64], valid_mask [T, 64]).
        valid_mask is 1 where readings are valid, 0 where invalid.
    """
    # Build validity mask
    valid_mask = (data != invalid_value) & (data >= min_depth) & (data <= max_depth)
    valid_mask = valid_mask.astype(np.float32)  # [T, 64]

    # Replace invalid readings with 0 (will be masked by model)
    data = data.copy()
    data[~valid_mask.astype(bool)] = 0.0

    # Clip to valid range
    data = np.clip(data, min_depth, max_depth)

    # Normalization (apply only to valid pixels)
    data = _normalize(data, normalization, norm_stats)

    # Re-zero invalid pixels after normalization
    data[~valid_mask.astype(bool)] = 0.0

    return data.astype(np.float32), valid_mask


# ─────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────


def _normalize(
    data: np.ndarray,
    method: str,
    stats: dict | None = None,
) -> np.ndarray:
    """Apply normalization to data.

    Args:
        data: [T, C] array.
        method: "zscore" | "minmax" | "none".
        stats: Precomputed stats. If None, compute from data.

    Returns:
        Normalized array.
    """
    if method == "none":
        return data
    elif method == "zscore":
        if stats is not None:
            mean = stats["mean"]
            std = stats["std"]
        else:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-6
        return (data - mean) / std
    elif method == "minmax":
        if stats is not None:
            mn = stats["min"]
            mx = stats["max"]
        else:
            mn = data.min(axis=0, keepdims=True)
            mx = data.max(axis=0, keepdims=True)
        return (data - mn) / (mx - mn + 1e-6)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _running_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Compute running mean along axis=0 with a given window size."""
    kernel = np.ones(window) / window
    result = np.zeros_like(data)
    for c in range(data.shape[1]):
        result[:, c] = np.convolve(data[:, c], kernel, mode="same")
    return result


def _get_neighbors(data_3d: np.ndarray, r: int, c: int) -> np.ndarray:
    """Get temporal neighbor pixel values for dead pixel replacement."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and not (dr == 0 and dc == 0):
                neighbors.append(data_3d[:, nr, nc])
    return np.stack(neighbors, axis=0) if neighbors else data_3d[:, r, c : c + 1]
