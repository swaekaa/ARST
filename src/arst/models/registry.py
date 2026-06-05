"""
Model registry for ARST.

Provides a central ``get_model(name, **kwargs)`` function that returns
a configured model instance without if-else chains in training scripts.

Usage::

    from arst.models.registry import get_model

    model = get_model("mlp", num_classes=4)
    model = get_model("cnn", active_modalities=["imu", "tof"])
    model = get_model("transformer", d_model=256, num_layers=4)

Registered models (Phase 2):
    - ``"majority"``     → :class:`~arst.models.baselines.majority.MajorityBaseline`
    - ``"random"``       → :class:`~arst.models.baselines.majority.RandomBaseline`
    - ``"mlp"``          → :class:`~arst.models.baselines.mlp.MLPBaseline`
    - ``"cnn"``          → :class:`~arst.models.baselines.cnn.CNNBaseline`
    - ``"lstm"``         → :class:`~arst.models.baselines.lstm.LSTMBaseline`
    - ``"transformer"``  → :class:`~arst.models.baselines.transformer.TransformerBaseline`

Adding new models (Phase 3+):
    Register in :data:`_REGISTRY` below.  No other file needs to change.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from arst.models.baselines.cnn import CNNBaseline
from arst.models.baselines.lstm import LSTMBaseline
from arst.models.baselines.majority import MajorityBaseline, RandomBaseline
from arst.models.baselines.mlp import MLPBaseline
from arst.models.baselines.transformer import TransformerBaseline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Registry dict: name → class
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[nn.Module]] = {
    "majority": MajorityBaseline,
    "random": RandomBaseline,
    "mlp": MLPBaseline,
    "cnn": CNNBaseline,
    "lstm": LSTMBaseline,
    "transformer": TransformerBaseline,
    # Phase 3+ will add:
    # "arst": ARSTModel,
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """
    Instantiate a registered model by name.

    Args:
        name:     Case-insensitive model key (e.g. ``"mlp"``, ``"transformer"``).
        **kwargs: Constructor keyword arguments forwarded to the model class.

    Returns:
        Configured :class:`~torch.nn.Module` instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Model '{name}' not in registry. Available: {available}")
    cls = _REGISTRY[key]
    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Registry: instantiated '%s' with %s trainable params.", key, f"{n_params:,}")
    return model


def get_model_from_cfg(cfg: Any) -> nn.Module:
    """
    Instantiate a model from a Hydra/OmegaConf config object.

    The config must have a ``name`` field.  All other fields are forwarded
    as constructor kwargs.

    Args:
        cfg: OmegaConf DictConfig or plain dict with at least ``name`` key.

    Returns:
        Configured model instance.

    Example config (``configs/model/mlp.yaml``)::

        name: mlp
        num_classes: 4
        hidden_dims: [512, 256, 128]
        dropout: 0.3
        active_modalities: [imu, thermo, tof]
    """
    try:
        from omegaconf import OmegaConf

        if hasattr(cfg, "_metadata"):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    if isinstance(cfg, dict):
        name: str = cfg.pop("name")
        kwargs: dict[str, Any] = cfg
    else:
        name = cfg.name
        kwargs = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and k != "name"}

    return get_model(name, **kwargs)


def list_models() -> list[str]:
    """Return all registered model names sorted alphabetically."""
    return sorted(_REGISTRY)


def register_model(name: str, cls: type[nn.Module]) -> None:
    """
    Register a new model class under the given name.

    Args:
        name: Registry key (lowercase recommended).
        cls:  Model class (subclass of :class:`~torch.nn.Module`).
    """
    _REGISTRY[name.lower()] = cls
    logger.debug("Registry: registered '%s' → %s", name, cls.__name__)
