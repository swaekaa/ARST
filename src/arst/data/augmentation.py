"""
Online data augmentation transforms for ARST training.

All transforms operate on the sample dict:
    {"imu": np.ndarray [T,6], "thermo": np.ndarray [T,64],
     "tof": np.ndarray [T,64], "tof_mask": np.ndarray [T,64], "label": int}

Compose multiple transforms using augmentation.Compose([...]).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np


class Compose:
    """Compose a sequence of transforms."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomTimeWarp:
    """
    Apply random time warping to all sensor modalities simultaneously.

    Randomly stretches and compresses segments of the time axis,
    simulating natural variation in movement speed.
    """

    def __init__(self, sigma: float = 0.2, n_knots: int = 4, p: float = 0.5):
        self.sigma = sigma
        self.n_knots = n_knots
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        T = sample["imu"].shape[0]
        # Generate warp path
        knot_positions = np.linspace(0, T - 1, self.n_knots + 2)
        warp_factors = np.random.normal(1.0, self.sigma, size=self.n_knots + 2)
        warp_factors = np.clip(warp_factors, 0.5, 2.0)
        cumulative = np.cumsum(warp_factors)
        warp_path = np.interp(
            np.arange(T),
            knot_positions,
            cumulative / cumulative[-1] * (T - 1),
        )

        for key in ("imu", "thermo", "tof"):
            arr = sample[key]  # [T, C]
            warped = np.zeros_like(arr)
            for c in range(arr.shape[1]):
                warped[:, c] = np.interp(warp_path, np.arange(T), arr[:, c])
            sample[key] = warped

        return sample


class RandomChannelNoise:
    """Add Gaussian noise to each sensor channel independently."""

    def __init__(self, sigma: float = 0.05, p: float = 0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        for key in ("imu", "thermo", "tof"):
            noise = np.random.normal(0.0, self.sigma, size=sample[key].shape).astype(np.float32)
            sample[key] = sample[key] + noise

        return sample


class RandomModalityDrop:
    """
    Randomly zero-out one or more modalities to train robustness.

    Each modality is independently dropped with probability p_drop.
    Useful for Phase 6: Missing Modality Robustness training.
    """

    def __init__(self, p_drop: float = 0.2):
        self.p_drop = p_drop

    def __call__(self, sample: dict) -> dict:
        for key in ("imu", "thermo", "tof"):
            if random.random() < self.p_drop:
                sample[key] = np.zeros_like(sample[key])

        return sample


class RandomTimeShift:
    """
    Shift the time axis by a random offset (circular or zero-padded).

    Simulates timing jitter between recording starts.
    """

    def __init__(self, max_shift: int = 20, mode: str = "zero", p: float = 0.5):
        self.max_shift = max_shift
        self.mode = mode  # "zero" | "circular"
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        shift = random.randint(-self.max_shift, self.max_shift)

        for key in ("imu", "thermo", "tof"):
            arr = sample[key]  # [T, C]
            T = arr.shape[0]

            if self.mode == "circular":
                sample[key] = np.roll(arr, shift, axis=0)
            else:  # zero padding
                shifted = np.zeros_like(arr)
                if shift > 0:
                    shifted[shift:] = arr[: T - shift]
                elif shift < 0:
                    shifted[: T + shift] = arr[-shift:]
                else:
                    shifted = arr
                sample[key] = shifted

        return sample


class MagnitudeScaling:
    """Scale signal amplitude by a random factor per channel."""

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        for key in ("imu", "thermo", "tof"):
            arr = sample[key]  # [T, C]
            scale = np.random.uniform(*self.scale_range, size=(1, arr.shape[1])).astype(np.float32)
            sample[key] = arr * scale

        return sample


def build_augmentation_pipeline(config: dict, is_training: bool) -> Compose:
    """
    Build augmentation pipeline from a config dict.

    Args:
        config: Augmentation config (from Hydra).
        is_training: If False, return identity (no augmentation at eval time).

    Returns:
        Compose transform.
    """
    if not is_training or not config.get("enabled", True):
        return Compose([])

    transforms = []

    if config.get("random_time_warp", {}).get("enabled", False):
        transforms.append(RandomTimeWarp(sigma=config["random_time_warp"].get("sigma", 0.2)))

    if config.get("random_channel_noise", {}).get("enabled", False):
        transforms.append(
            RandomChannelNoise(sigma=config["random_channel_noise"].get("sigma", 0.05))
        )

    if config.get("random_modality_drop", {}).get("enabled", False):
        transforms.append(
            RandomModalityDrop(p_drop=config["random_modality_drop"].get("p_drop", 0.2))
        )

    if config.get("random_time_shift", {}).get("enabled", False):
        transforms.append(
            RandomTimeShift(max_shift=config["random_time_shift"].get("max_shift", 20))
        )

    if config.get("magnitude_scaling", {}).get("enabled", False):
        transforms.append(
            MagnitudeScaling(scale_range=config["magnitude_scaling"].get("range", [0.9, 1.1]))
        )

    return Compose(transforms)
