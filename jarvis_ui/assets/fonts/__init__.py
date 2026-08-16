"""Font lookup helper.

Tries to load a custom sci-fi font from this folder (any ``*.ttf``/``*.otf``
file is registered automatically). If none is found we fall back to a
monospace font (Consolas / Courier New) so the UI always renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_FONT_DIR = Path(__file__).resolve().parent
_REGISTERED = False


def _ensure_registered() -> None:
    """Register every font file shipped in this directory exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from PySide6.QtGui import QFontDatabase
    except Exception:
        return
    for font_file in _FONT_DIR.glob("*.[ot]tf"):
        QFontDatabase.addApplicationFont(str(font_file))
    _REGISTERED = True


def pick_font_family(preferred: Optional[str] = None) -> str:
    """Return the best available monospace / sci-fi font family name.

    Parameters
    ----------
    preferred:
        Optional family name to try first (e.g. "Orbitron").
    """
    _ensure_registered()
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [
        "Consolas",
        "Courier New",
        "Lucida Console",
        "Menlo",
        "DejaVu Sans Mono",
        "Liberation Mono",
        "monospace",
    ]
    try:
        from PySide6.QtGui import QFontDatabase

        families = set(QFontDatabase().families())
    except Exception:
        families = set()
    for family in candidates:
        if family in families:
            return family
    return "Courier New"