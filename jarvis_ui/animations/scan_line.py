"""Scanning line + holographic glitch overlays.

Scan line
---------
A thin horizontal glowing line sweeps top-to-bottom across the whole UI every
four seconds, like a radar scan.  Pure decoration, ignores mouse events.

Glitch effect
-------------
Every 30–60 seconds the overlay briefly "glitches": random horizontal slices
shift a few pixels and flicker, then stabilise within ~200ms.  The offset is
exposed via ``state.glitch_active`` / ``state.glitch_seed`` so other widgets
(such as the sphere camera) can react subtly too.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math
import random
import time

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

# ── Colours (configurable) ────────────────────────────────────────────────────
SCAN_COLOR = (0.0, 0.9, 1.0, 0.5)
GLITCH_CYAN = (0.0, 0.95, 1.0, 0.75)
GLITCH_RED = (1.0, 0.2, 0.2, 0.6)
GLITCH_GREEN = (0.0, 1.0, 0.6, 0.4)

SCAN_PERIOD = 4.0
GLITCH_MIN_INTERVAL = 30.0
GLITCH_MAX_INTERVAL = 60.0
GLITCH_DURATION = 0.2


class ScanLineOverlay(QWidget):
    """Full-window scanning line, transparent to mouse events."""

    def __init__(self, parent=None, state=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._t = 0.0

    def tick(self, dt: float) -> None:
        self._t += dt
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

        w, h = self.width(), self.height()
        phase = self._t % SCAN_PERIOD
        # Sweep during the first 2.6s of each period, then pause.
        if phase > 2.6:
            painter.end()
            return

        frac = phase / 2.6
        y = frac * h

        alpha = 120 * (1.0 - abs(frac - 0.5) * 1.4)
        alpha = max(20, int(alpha))
        color = QColor(*[int(c * 255) for c in SCAN_COLOR[:3]], alpha)

        # Glow band.
        grad_alpha = int(alpha * 0.35)
        pen = QPen(QColor(color.red(), color.green(), color.blue(), grad_alpha), 10)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, y), QPointF(w, y))

        # Core line.
        pen = QPen(QColor(color.red(), color.green(), color.blue(), alpha), 1)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, y), QPointF(w, y))

        painter.end()


class GlitchOverlay(QWidget):
    """Holographic glitch effect that fires periodically for ~200ms."""

    def __init__(self, parent=None, state=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._next_trigger = time.monotonic() + random.uniform(GLITCH_MIN_INTERVAL, GLITCH_MAX_INTERVAL)
        self._active_until = 0.0
        self._slices: list[tuple[int, int, int]] = []  # (y, height, type)

    def tick(self, dt: float) -> None:
        now = time.monotonic()
        if now >= self._next_trigger:
            self._active_until = now + GLITCH_DURATION
            self._next_trigger = now + random.uniform(GLITCH_MIN_INTERVAL, GLITCH_MAX_INTERVAL)
            self._slices = [
                (random.randint(0, max(1, self.height() - 4)), random.randint(2, 8), random.randint(0, 2))
                for _ in range(random.randint(6, 14))
            ]
            if self.state is not None:
                self.state.glitch_active = True
                self.state.glitch_seed = random.random()
        if now < self._active_until:
            self.update()
        elif self.state is not None and self.state.glitch_active:
            self.state.glitch_active = False

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

        now = time.monotonic()
        if now >= self._active_until or not self._slices:
            painter.end()
            return

        w, h = self.width(), self.height()
        for y, slice_h, kind in self._slices:
            if y + slice_h > h:
                continue
            if kind == 0:
                color = GLITCH_CYAN
            elif kind == 1:
                color = GLITCH_RED
            else:
                color = GLITCH_GREEN
            offset = random.randint(-4, 4)
            alpha = int(color[3] * 255)
            painter.fillRect(QRectF(offset, y, w - abs(offset), slice_h), QColor(*[int(c * 255) for c in color[:3]], alpha))

        painter.end()