"""Voice Waveform Visualizer.

A circular ring of vertical bars that grow/shrink with audio amplitude, plus a
classic horizontal scrolling waveform below it.

Behaviour
---------
* Quiet          — flat minimal circle.
* User speaking  — bars explode outward; colour shifts cyan -> orange.
* JARVIS speaking— bars driven by the same audio signal but tinted blue.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from jarvis_ui.ui_state import LISTENING, SPEAKING, IDLE

# ── Colours (configurable) ────────────────────────────────────────────────────
BAR_QUIET = (0.0, 0.83, 1.0, 1.0)        # cyan
BAR_LOUD = (1.0, 0.42, 0.0, 1.0)         # iron-man orange
BAR_SPEAK = (0.3, 0.6, 1.0, 1.0)         # blue while JARVIS speaks
BAR_BG = (0.0, 0.35, 0.7, 0.28)          # resting bars
WAVE_COLOR = (0.0, 0.85, 1.0, 0.9)       # horizontal waveform line
WAVE_SPEAK = (0.35, 0.7, 1.0, 0.95)      # horizontal waveform while speaking
GRID_COLOR = (0.0, 0.2, 0.5, 0.12)       # faint center circle

N_BARS = 56
CIRCLE_RADIUS = 62.0
BAR_MIN = 4.0
BAR_MAX = 48.0
WAVE_HISTORY = 220


def _lerp_rgb(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class WaveformWidget(QWidget):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.state = state
        self.setMinimumHeight(120)
        self._history: list[float] = [0.0] * WAVE_HISTORY
        self._level = 0.0
        self._status = IDLE
        self._smooth = 0.0

    def update_frame(self, dt: float, status: str, level: float) -> None:
        self._status = status
        target = level if status in (LISTENING, SPEAKING) else 0.0
        # Attack fast, release slow → feels "alive".
        self._smooth += (target - self._smooth) * (8.0 * dt if target > self._smooth else 2.2 * dt)
        self._level = max(0.0, min(1.0, self._smooth))
        self._history.append(self._level)
        if len(self._history) > WAVE_HISTORY:
            self._history = self._history[-WAVE_HISTORY:]
        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h * 0.42

        speaking = self._status == SPEAKING
        color_scheme = BAR_SPEAK if speaking else None

        # Faint baseline circle.
        pen = QPen(QColor(*[int(c * 255) for c in GRID_COLOR]))
        painter.setPen(pen)
        painter.drawEllipse(QPointF(cx, cy), CIRCLE_RADIUS, CIRCLE_RADIUS)

        # Circular bars.
        for i in range(N_BARS):
            angle = 2.0 * math.pi * i / N_BARS - math.pi / 2
            ca, sa = math.cos(angle), math.sin(angle)

            # Pseudo-frequency bands: higher indexes get livelier.
            band = 0.35 + 0.65 * abs(math.sin(i * 0.35))
            amp = max(0.0, self._level * band - 0.05)
            length = BAR_MIN + amp * BAR_MAX

            if speaking:
                r, g, b = _lerp_rgb((0.0, 0.2, 0.4), color_scheme, self._level)
            else:
                r, g, b = _lerp_rgb(BAR_QUIET, BAR_LOUD, self._level)

            x0 = cx + ca * (CIRCLE_RADIUS - BAR_MIN)
            y0 = cy + sa * (CIRCLE_RADIUS - BAR_MIN)
            x1 = cx + ca * (CIRCLE_RADIUS + length)
            y1 = cy + sa * (CIRCLE_RADIUS + length)

            alpha = 40 + int(215 * amp)
            pen = QPen(QColor(r, g, b, alpha))
            pen.setWidth(2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Horizontal waveform below.
        wave_y = h * 0.78
        wave_color = WAVE_SPEAK if speaking else WAVE_COLOR
        path = np.linspace(0, w, WAVE_HISTORY)
        samples = np.asarray(self._history, dtype=np.float64)

        # Glow pass (thick, translucent).
        pen = QPen(QColor(*[int(c * 255) for c in wave_color[:3]], 60))
        pen.setWidth(6)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        for x, s in zip(path, samples):
            if x < 1:
                continue
            y = wave_y - s * 26.0
            painter.drawPoint(QPointF(x, y))

        # Core pass (thin, bright).
        pen = QPen(QColor(*[int(c * 255) for c in wave_color[:3]], 235))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        points = [QPointF(float(x), wave_y - float(s) * 26.0) for x, s in zip(path, samples)]
        if points:
            painter.drawPolyline(points)

        # Center baseline.
        pen = QPen(QColor(0, 60, 120, 90))
        painter.setPen(pen)
        painter.drawLine(QPointF(0, wave_y), QPointF(w, wave_y))

        painter.end()