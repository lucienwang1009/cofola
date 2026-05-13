"""Shared logging configuration for cofola."""
from __future__ import annotations

import sys

from loguru import logger


def setup_logging(debug: bool) -> None:
    """Configure logging for cofola.

    Sets up loguru handlers for cofola modules at the requested level.
    WFOMC uses logzero internally and is always kept at INFO level to
    avoid leaking WFOMC debug output when cofola debug is enabled.

    Args:
        debug: If True, enable DEBUG level for cofola modules.
    """
    # Configure loguru for cofola
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level,
               filter=lambda r: r["name"].startswith("cofola"))
    # non-cofola, non-wfomc (e.g. __main__ in scripts) always at INFO
    logger.add(sys.stderr, level="INFO",
               filter=lambda r: not r["name"].startswith("cofola")
                                and not r["name"].startswith("wfomc"))
