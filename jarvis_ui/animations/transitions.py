"""Transition effects: screen fade / flash utilities.

Provides a small :class:`FadeLayer` overlay that can fade to black (for a
"power down") or flash white/cyan (for the boot power-up), plus an easing
helper.  Used by the boot sequence and on shutdown.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget

# ── Colours (configurable) ────────────────────────────────────────────────────
FADE_COLOR = (0.0, 0.0, 0.0, 1.0)
FLASH_COLOR = (0.0, 0.85, 1.0, 1.0)


def ease_in_out(t: float) -> float:
    """Smoothstep easing in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class FadeLayer(QWidget):
    """Covers the window with a fading colour overlay.

    Supports two modes:

    * ``fade``  — alpha goes 0 → 1 (to black) or 1 → 0 (reveal).
    * ``flash`` — a brief bright pulse that decays (power-up effect).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._mode = "none"
        self._start = 0.0
        self._duration = 1.0
        self._alpha = 0.0
        self._flash_alpha = 0.0
        self._on_done = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._tick_rate = 1 / 60.0

    # ── Public API ─────────────────────────────────────────────────────────

    def fade_to_black(self, duration: float = 0.8, on_done=None) -> None:
        self._mode = "fade_in"  # alpha 0 -> 1
        self._start = time.monotonic()
        self._duration = duration
        self._on_done = on_done
        self._alpha = 0.0
        self.show()
        self._timer.start(16)

    def fade_in(self, duration: float = 0.8, on_done=None) -> None:
        self._mode = "fade_out"  # alpha 1 -> 0
        self._start = time.monotonic()
        self._duration = duration
        self._on_done = on_done
        self._alpha = 1.0
        self.show()
        self._timer.start(16)

    def flash(self, duration: float = 0.4, on_done=None) -> None:
        self._mode = "flash"
        self._start = time.monotonic()
        self._duration = duration
        self._on_done = on_done
        self._flash_alpha = 1.0
        self.show()
        self._timer.start(16)

    def stop(self) -> None:
        self._mode = "none"
        self._timer.stop()
        self.hide()

    # ── Internals ──────────────────────────────────────────────────────────

    def _step(self) -> None:
        now = time.monotonic()
        t = (now - self._start) / self._duration
        t = max(0.0, min(1.0, t))

        if self._mode == "fade_in":
            self._alpha = ease_in_out(t)
            if t >= 1.0:
                self._finish()
        elif self._mode == "fade_out":
            self._alpha = 1.0 - ease_in_out(t)
            if t >= 1.0:
                self._finish()
        elif self._mode == "flash":
            self._flash_alpha = max(0.0, 1.0 - ease_in_out(t))
            if t >= 1.0:
                self._finish()
        self.update()

    def _finish(self) -> None:
        self._timer.stop()
        self.hide()
        self._alpha = 0.0
        self._flash_alpha = 0.0
        if self._on_done:
            cb = self._on_done
            self._on_done = None
            cb()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._mode == "flash" and self._flash_alpha > 0:
            alpha = int(self._flash_alpha * 255)
            painter.fillRect(self.rect(), QColor(*[int(c * 255) for c in FLASH_COLOR[:3]], alpha))
        elif self._mode in ("fade_in", "fade_out") and self._alpha > 0:
            alpha = int(self._alpha * 255)
            painter.fillRect(self.rect(), QColor(*[int(c * 255) for c in FADE_COLOR[:3]], alpha))
        painter.end()