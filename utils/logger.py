"""
utils/logger.py — Logging for JARVIS

Design goals:
  * Normal operation prints a CLEAN console: only WARNING/ERROR (and a
    few INFO milestones) show up. DEBUG noise appears only with
    `jarvis --debug` (or JARVIS_DEBUG=1).
  * Everything is always written to jarvis.log at DEBUG level so a
    problem can be diagnosed without running in debug mode.
  * All module loggers share one console + one file handler (attached
    to the root logger), so `set_debug()` switches the whole app at once.
"""

import logging
import sys
from pathlib import Path

# Console shows only WARNING/ERROR by default so normal operation stays
# clean (the user-facing status lines are printed directly by main.py).
# `jarvis --debug` or LOG_LEVEL=DEBUG flips the console to DEBUG;
# jarvis.log always records everything at DEBUG.
_CONSOLE_LEVEL = logging.WARNING

try:
    from config import log_config

    if (log_config.LEVEL or "INFO").strip().upper() == "DEBUG":
        _CONSOLE_LEVEL = logging.DEBUG
except Exception:
    pass

# Keep track of the console handler so set_debug() can adjust it.
_console_handler: logging.Handler | None = None


def _configure_root() -> None:
    """Attach the shared console + file handlers to the root logger once."""
    global _console_handler
    root = logging.getLogger()
    if getattr(root, "_jarvis_configured", False):
        return

    root.setLevel(logging.DEBUG)

    # ── Console handler (INFO normally, DEBUG with --debug) ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_CONSOLE_LEVEL)
    console.setFormatter(
        logging.Formatter(fmt="[%(levelname)s] %(message)s")
    )
    root.addHandler(console)
    _console_handler = console

    # ── File handler (always DEBUG) ──
    try:
        log_path = Path(__file__).parent.parent / "jarvis.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] "
                    "%(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
    except Exception:
        pass  # File logging is optional

    root._jarvis_configured = True  # type: ignore[attr-defined]


def set_debug(enabled: bool = True) -> None:
    """Show DEBUG-level output on the console (used by `jarvis --debug`)."""
    global _CONSOLE_LEVEL
    _CONSOLE_LEVEL = logging.DEBUG if enabled else logging.INFO
    _configure_root()
    if _console_handler is not None:
        _console_handler.setLevel(_CONSOLE_LEVEL)
    logging.getLogger().debug("Debug logging enabled.")


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger with console + file output.

    Call this from any module: logger = get_logger(__name__)
    """
    _configure_root()
    return logging.getLogger(name)


# Keep these for any code that imports them directly.
def get_info_logger(name: str) -> logging.Logger:
    return get_logger(name)


def get_debug_logger(name: str) -> logging.Logger:
    return get_logger(name)
