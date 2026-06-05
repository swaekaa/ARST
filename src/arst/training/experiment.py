"""
Experiment context for ARST Phase 2.

Centralises all reproducibility and path-management concerns that the Trainer
needs but that are logically separate from the training loop itself:

    - PRNG seeding (Python / NumPy / PyTorch / CUDA)
    - Device selection
    - Experiment directory creation (``experiments/<run_id>/``)
    - Run-ID generation
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Set PRNG seeds for full reproducibility.

    Args:
        seed:          Integer seed shared across all libraries.
        deterministic: If ``True``, enable ``torch.use_deterministic_algorithms``
                       and set the CUBLAS workspace deterministic env var.
                       This may reduce GPU performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("Deterministic mode enabled (slower but reproducible).")

    logger.info("Seeded: seed=%d, deterministic=%s", seed, deterministic)


def get_device(device_str: str | None = None) -> torch.device:
    """
    Resolve the best available device.

    Args:
        device_str: Explicit device string (``"cuda"``, ``"cpu"``, ``"mps"``).
                    If ``None``, auto-selects CUDA > MPS > CPU.

    Returns:
        :class:`torch.device`.
    """
    # ── Startup diagnostics ──────────────────────────────────────────────
    logger.info("PyTorch version  : %s", torch.__version__)
    logger.info("CUDA version     : %s", torch.version.cuda or "N/A (CPU-only build)")
    logger.info("CUDA available   : %s", torch.cuda.is_available())
    logger.info("Device count     : %d", torch.cuda.device_count())

    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU              : %s  (%.1f GB VRAM)", gpu_name, vram_gb)
        logger.info("Selected device  : cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Selected device  : Apple MPS")
    else:
        device = torch.device("cpu")
        _cuda_build = torch.version.cuda
        if _cuda_build is None:
            # CPU-only PyTorch installed -- provide exact fix
            import sys

            _msg = (
                "\n"
                "========================================================\n"
                "  WARNING: CPU-only PyTorch detected!\n"
                "  NVIDIA GPU is present but torch.cuda.is_available()\n"
                "  returns False because a CPU-only wheel is installed.\n"
                "\n"
                "  Fix: reinstall PyTorch with CUDA 12.1 support:\n"
                "\n"
                "    pip uninstall torch torchvision torchaudio -y\n"
                "    pip install torch torchvision torchaudio \\\n"
                "        --index-url https://download.pytorch.org/whl/cu121\n"
                "\n"
                '  Then verify with: python -c "import torch; print(torch.cuda.is_available())"\n'
                "========================================================\n"
            )
            print(_msg, file=sys.stderr, flush=True)
            logger.warning(_msg)
        logger.warning("Selected device  : CPU -- training will be slow.")

    return device


def make_run_id(model_name: str, seed: int) -> str:
    """
    Generate a unique run identifier.

    Format: ``<model_name>_seed<seed>_<YYYYMMDD_HHMMSS>``

    Args:
        model_name: Model/experiment name.
        seed:       Random seed used for this run.

    Returns:
        Run ID string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_seed{seed}_{timestamp}"


class ExperimentContext:
    """
    Bundles all experiment-level metadata and filesystem paths.

    Args:
        run_id:       Unique run identifier (see :func:`make_run_id`).
        base_dir:     Root experiments directory (default: ``experiments/``).
        seed:         Random seed.
        deterministic: Enable fully deterministic PyTorch ops.
        device_str:   Explicit device string or ``None`` for auto-detect.
    """

    def __init__(
        self,
        run_id: str,
        base_dir: str | Path = "experiments",
        seed: int = 42,
        deterministic: bool = False,
        device_str: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.deterministic = deterministic

        # Seed first, then resolve device
        seed_everything(seed, deterministic)
        self.device = get_device(device_str)

        # File paths
        self.run_dir = self.base_dir / run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.output_dir = self.run_dir / "outputs"

        for d in (self.checkpoint_dir, self.log_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ExperimentContext: run_id=%s  seed=%d  device=%s  run_dir=%s",
            run_id,
            seed,
            self.device,
            self.run_dir,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return serialisable context metadata."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": str(self.device),
            "run_dir": str(self.run_dir),
        }
