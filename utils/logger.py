"""Lightweight console logging with optional ANSI colours.

The API is intentionally tiny — every module calls ``info``, ``status``,
``ok``, ``warning`` and ``error``. The ``set_sink`` hook lets a UI capture
all output, and ``tick`` / ``report`` provide cheap timing for diagnostics.

ANSI colours are enabled on Windows by flipping on VT processing in the
console (a no-op elsewhere). Colours degrade gracefully on dumb terminals.
"""

import ctypes
import os
import sys
import time
from typing import Any, Callable

# ── ANSI support ──────────────────────────────────────────────────────────────

# Windows consoles need this one-time call before they honour ANSI escapes.
if os.name == "nt":
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(
            kernel32.GetStdHandle(-11),  # STD_OUTPUT_HANDLE
            kernel32.GetConsoleMode(kernel32.GetStdHandle(-11)) | 0x0004,
        )
    except Exception:
        pass


def _supports_colour() -> bool:
    """Best-effort: True when the console is likely to render ANSI codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.name == "nt":
        return True  # VT processing enabled above
    return bool(sys.stdout.isatty())


_COLOUR = _supports_colour()

# ANSI codes (only applied when _COLOUR is True).
_RESET = "\033[0m"
_CODES = {
    "info": "\033[36m",     # cyan
    "status": "\033[33m",   # yellow
    "ok": "\033[32m",       # green
    "warning": "\033[93m",  # bright yellow
    "error": "\033[91m",    # bright red
}

_sink: Callable[[str, Any], None] | None = None


def set_sink(sink: Callable[[str, Any], None] | None) -> None:
    """Redirect all logger output to *sink* (or restore printing with None)."""
    global _sink
    _sink = sink


def _emit(level: str, msg: Any) -> None:
    if _sink is not None:
        _sink(level, msg)
        return
    text = str(msg)
    if _COLOUR and level in _CODES:
        text = f"{_CODES[level]}{text}{_RESET}"
    print(f"[{level}] {text}")


def info(msg: Any) -> None:
    _emit("info", msg)


def status(msg: Any) -> None:
    _emit("status", msg)


def ok(msg: Any) -> None:
    _emit("ok", msg)


def warning(msg: Any) -> None:
    _emit("warning", msg)


def error(msg: Any) -> None:
    _emit("error", msg)


def tick() -> float:
    """Return a high-resolution timestamp (for measuring elapsed time)."""
    return time.perf_counter()


def report(label: str, start: float) -> float:
    """Return seconds since *start* and notify the sink (or print)."""
    elapsed = time.perf_counter() - start
    if _sink is not None:
        _sink("report", (label, elapsed))
    else:
        print(f"[{label}] {elapsed:.2f} sec")
    return elapsed