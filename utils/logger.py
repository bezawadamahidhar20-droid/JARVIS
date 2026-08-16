"""
utils/logger.py — Logging configuration for JARVIS.
 
[FIX m5] Added __all__ exports.
"""
 
import logging
import sys
from pathlib import Path

__all__ = [
    "get_logger",
    "set_debug",
]

# Track if we're in debug mode
_debug_mode = False

# Project root for log file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Guard so the root logger is configured only once
_ROOT_CONFIGURED = False


def _configure_root() -> None:
    """Configure the root logger with a console + rotating file handler."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    _ROOT_CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(console)

    try:
        from logging.handlers import RotatingFileHandler
        from config import log_config
        log_file = _PROJECT_ROOT / log_config.FILE
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            str(log_file), maxBytes=10 * 1024 * 1024,
            backupCount=5, encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root.addHandler(file_handler)
    except Exception:
        pass  # File logging optional


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    _configure_root()
    logger = logging.getLogger(f"jarvis.{name}")
    return logger
 
 
def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug logging globally."""
    global _debug_mode
    _debug_mode = enabled
    
    level = logging.DEBUG if enabled else logging.INFO
    
    # Update all jarvis loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("jarvis."):
            logging.getLogger(name).setLevel(level)
 