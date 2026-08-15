"""
utils/logger.py — Simple logger for JARVIS
Works with Python 3.14
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with console output.
    Call this from any module: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Don't add duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Format: [level] message
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Try to add file handler
    try:
        log_path = Path(__file__).parent.parent / "jarvis.log"
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # File logging optional

    return logger


# Keep these for any code that imports them directly
def get_info_logger(name: str) -> logging.Logger:
    return get_logger(name)


def get_debug_logger(name: str) -> logging.Logger:
    return get_logger(name)