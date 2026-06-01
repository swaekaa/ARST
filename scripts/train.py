"""
Main training script for ARST.

Usage:
    python scripts/train.py --config-name arst_full experiment=arst/arst_full_v1
    python scripts/train.py --config-name baseline_transformer experiment=baseline/transformer

Powered by Hydra for config composition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from arst.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Construct model from config."""
    model_type = cfg.model.type

    if model_type == "arst":
        from arst.models.arst import ARSTModel

        model = ARSTModel(
            d_model=cfg.model.d_model,
            num_classes=cfg.model.classification_head.num_classes,
            imu_config=OmegaConf.to_container(cfg.model.imu_encoder, resolve=True),
            thermal_config=OmegaConf.to_container(cfg.model.thermal_encoder, resolve=True),
            tof_config=OmegaConf.to_container(cfg.model.tof_encoder, resolve=True),
            reliability_config=OmegaConf.to_container(cfg.model.reliability, resolve=True),
            fusion_config=OmegaConf.to_container(cfg.model.fusion, resolve=True),
            head_config=OmegaConf.to_container(cfg.model.classification_head, resolve=True),
        )
    elif model_type == "baseline_transformer":
        from arst.models.baselines.transformer import TransformerBaseline

        model = TransformerBaseline(
            d_model=cfg.model.d_model,
            num_classes=cfg.model.classification_head.num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(
        "Built model: %s (%d parameters)",
        model_type,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = cfg.training.optimizer
    if opt_cfg.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.type}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    """Build LR scheduler from config."""
    sched_cfg = cfg.training.scheduler
    if sched_cfg.type == "cosine_with_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=sched_cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs - sched_cfg.warmup_epochs,
            eta_min=sched_cfg.min_lr,
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[sched_cfg.warmup_epochs]
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_cfg.type}")


@hydra.main(version_base=None, config_path="../configs", config_name="arst_full")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    setup_logging()
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- W&B Initialization ---
    if cfg.training.wandb.enabled:
        run = wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.get("experiment", {}).get("name", None),
            tags=cfg.get("experiment", {}).get("tags", []),
            group=(
                cfg.get("experiment", {}).get("tags", [""])[0]
                if cfg.get("experiment", {}).get("tags")
                else None
            ),
        )

    # --- Data ---
    # TODO: Initialize ARSTDataModule from cfg.data
    logger.info("Data pipeline: pending dataset download (run scripts/download_data.py first)")

    # --- Model ---
    model = build_model(cfg).to(device)

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg.training))
    logger.info("Model ready. Starting training loop...")

    # TODO: Training loop implementation in arst.training.trainer


def entry_point():
    main()


if __name__ == "__main__":
    entry_point()
