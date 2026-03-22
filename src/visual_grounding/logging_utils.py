"""Loguru helpers for experiment tracking."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from loguru import logger

_LOGGER_INITIALIZED = False
_FILE_SINK_ID: int | None = None


def configure_logging(log_dir: Path, run_name: str | None = None) -> Path:
    """Set up stdout + file logging and return the file path."""
    global _LOGGER_INITIALIZED, _FILE_SINK_ID

    log_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"train_{run_name}.log"

    if not _LOGGER_INITIALIZED:
        logger.remove()
        logger.add(sys.stdout, level="INFO", colorize=True)
        _LOGGER_INITIALIZED = True

    if _FILE_SINK_ID is not None:
        logger.remove(_FILE_SINK_ID)
    _FILE_SINK_ID = logger.add(log_path, level="INFO", enqueue=True)

    logger.info("Logging to {log_path}", log_path=log_path)
    return log_path


__all__ = ["configure_logging"]
