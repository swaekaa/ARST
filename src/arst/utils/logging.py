"""
Logging utilities for ARST.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with Rich formatting.

    Args:
        name: Logger name (use __name__ in module files).
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """
    Set up root logger with optional file output.

    Args:
        log_dir: If provided, save logs to {log_dir}/run.log.
        level: Root logging level.
    """
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=True),
    ]

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
