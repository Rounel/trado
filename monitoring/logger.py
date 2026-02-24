"""
monitoring/logger.py — Configuration centralisée de Loguru.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


def setup_logger(settings: "Settings") -> None:
    """Configure Loguru selon les paramètres de l'application."""
    logger.remove()  # Supprime les handlers par défaut

    # Console (stderr)
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
               "<level>{message}</level>",
        colorize=True,
    )

    # Fichier rotatif
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        enqueue=True,  # thread-safe
    )

    logger.info(f"Logger initialisé — level={settings.log_level}  file={settings.log_file}")
