"""Logging helpers for CLI-style entry points and services."""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger once for CLI-oriented workflows.

    The helper is idempotent: repeated calls only adjust the root level instead
    of stacking duplicate handlers.
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(name)
