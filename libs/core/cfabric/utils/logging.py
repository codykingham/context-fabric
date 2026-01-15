"""
# Logging configuration for cfabric.

This module provides logging level constants and configuration utilities
for the cfabric library. Uses standard Python logging with output to stderr.
"""

from __future__ import annotations

import logging
import sys

# Verbosity level constants
VERBOSE = "verbose"
"""Show all messages including debug."""

AUTO = "auto"
"""Show info, warning, and error messages."""

TERSE = "terse"
"""Show only warning and error messages."""

DEEP = "deep"
"""Show only error messages (silent mode)."""

SILENT_D = AUTO
"""Default verbosity level."""

# Map verbosity levels to Python logging levels
LEVEL_MAP = {
    VERBOSE: logging.DEBUG,
    AUTO: logging.INFO,
    TERSE: logging.WARNING,
    DEEP: logging.ERROR,
}


def silentConvert(arg: str | bool | None) -> str:
    """Convert silent parameter to canonical string form.

    Accepts str, bool, or None and always returns a valid silent level string.
    """
    if arg is None:
        return SILENT_D
    if arg is False:
        return VERBOSE
    if arg is True:
        return DEEP
    if type(arg) is str and arg in {VERBOSE, AUTO, TERSE, DEEP}:
        return arg
    return SILENT_D


def configure_logging(silent: str = SILENT_D) -> None:
    """Configure the cfabric logger.

    Sets up the root cfabric logger with a stderr handler.

    Parameters
    ----------
    silent : str
        Verbosity level: "verbose", "auto", "terse", or "deep"
    """
    silent = silentConvert(silent)
    level = LEVEL_MAP.get(silent, logging.INFO)

    cfabric_logger = logging.getLogger("cfabric")

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not cfabric_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        cfabric_logger.addHandler(handler)
        # Prevent propagation to root logger (avoids duplicate messages
        # when other libraries like FastMCP configure root logging)
        cfabric_logger.propagate = False

    cfabric_logger.setLevel(level)


def set_logging_level(silent: str = SILENT_D) -> None:
    """Set the logging level for cfabric.

    Parameters
    ----------
    silent : str
        Verbosity level: "verbose", "auto", "terse", or "deep"
    """
    silent = silentConvert(silent)
    level = LEVEL_MAP.get(silent, logging.INFO)
    logging.getLogger("cfabric").setLevel(level)
