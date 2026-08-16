"""Boot sequence animation.

A full-screen overlay that plays when JARVIS starts:

1. Black screen.
2. "INITIALIZING J.A.R.V.I.S..." types out.
3. Progress bars fill one by one:
       [########] SPEECH ENGINE... OK
       [########] NEURAL NETWORK... OK
       [########] OLLAMA CONNECTION... OK
       [########] MICROPHONE... OK
       [########] MEMORY CORE... OK
4. The screen "powers up" with a flash.
5. The overlay fades out, revealing the main UI.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QWidget

# ── Colours (configurable) ────────────────────────────────────────────────────
BG = (0.0, 0.0, 0.0, 1.0)
TEXT = (0.0, 0.9, 1.0, 1.0)
BAR_BG = (0.0, 0.12, 0.25, 0.9)
BAR_FILL = (0.0, 0.9, 1.0, 1.0)
OK_COLOR = (0.0, 1.0, 0.53, 1.0)
FLASH_COLOR = (0.0, 0.85, 1.0, 1.0)

LINES = [
    "SPEECH ENGINE...",
    "NEURAL NETWORK...",
    "OLLAMA CONNECTION...",
    "MICROPHONE...",
    "MEMORY CORE...",
]

TITLE = "INITIALIZING J.A.R.V.I.S..."


class BootSequence(QWidget):
    """Runs the boot animation and calls ``on_finished`` at the end."""

    def __init__(self, parent=None, on_finished=None) -> None:
        super().__init__(parent)
        self.on_finished = on_finished
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._t = 0.0
        self._start = time.monotonic()
        self._title_chars = 0
        self._line_progress = 0.0      # 0..1 progress of the *current* bar
        self._current_line = 0
        self._phase = "bars"           # bars -> flash -> done
        self._flash = 0.0
        self._alpha = 1.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(16)

    def _step(self) -> None:
        dt = 1 / 60.0
        self._t += dt
        elapsed = time.monotonic() - self._start

        # Title types out over the first 1.2s.
        self._title_chars = min(len(TITLE), int(elapsed * 28))

        if self._phase == "bars":
            self._line_progress += dt * 0.9
            if self._line_progress >= 1.0:
                self._line_progress = 0.0
                self._current_line += 1
                if self._current_line >= len(LINES):
                    self._phase = "flash"
        elif self._phase == "flash":
            self._flash += dt * 3.0
            if self._flash >= 1.0:
                self._phase = "fade"
        elif self._phase == "fade":
            self._alpha -= dt * 2.2
            if self._alpha <= 0.0:
                self._alpha = 0.0
                self._timer.stop()
                self.hide()
                if self.on_finished:
                    self.on_finished()

        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(*[int(c * 255) for c in BG]))

        w, h = self.width(), self.height()
        cx = w / 2.0

        # Title.
        painter.setFont(QFont("Courier New", 22, QFont.Bold))
        painter.setPen(QColor(*[int(c * 255) for c in TEXT]))
        title = TITLE[: self._title_chars]
        painter.drawText(QRectF(0, h * 0.22, w, 40), Qt.AlignHCenter | Qt.AlignVCenter, title)

        # Progress lines.
        line_h = 26.0
        start_y = h * 0.30
        painter.setFont(QFont("Courier New", 11, QFont.Bold))
        for i, label in enumerate(LINES):
            y = start_y + i * line_h
            done = i < self._current_line
            is_current = i == self._current_line
            bar_w = w * 0.42
            bar_x = cx - bar_w / 2

            painter.setPen(QColor(*[int(c * 255) for c in TEXT]))
            painter.drawText(QRectF(bar_x - 190, y, 170, 20), Qt.AlignRight | Qt.AlignVCenter, label)

            # Bar background.
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(*[int(c * 255) for c in BAR_BG]))
            painter.drawRoundedRect(QRectF(bar_x, y, bar_w, 14), 4, 4)

            # Bar fill.
            fill = 1.0 if done else (self._line_progress if is_current else 0.0)
            if fill > 0.01:
                painter.setBrush(QColor(*[int(c * 255) for c in BAR_FILL]))
                painter.drawRoundedRect(QRectF(bar_x, y, bar_w * fill, 14), 4, 4)

            # Status.
            if done:
                painter.setPen(QColor(*[int(c * 255) for c in OK_COLOR]))
                painter.drawText(QRectF(bar_x + bar_w + 12, y, 40, 20), Qt.AlignLeft | Qt.AlignVCenter, "OK")
            elif is_current:
                painter.setPen(QColor(160, 200, 240, 200))
                painter.drawText(QRectF(bar_x + bar_w + 12, y, 80, 20), Qt.AlignLeft | Qt.AlignVCenter, "RUNNING")

        # Power-up flash.
        if self._phase == "flash" and self._flash < 1.0:
            alpha = int(255 * (1.0 - self._flash) * 0.9)
            painter.fillRect(self.rect(), QColor(*[int(c * 255) for c in FLASH_COLOR[:3]], alpha))
        elif self._phase == "fade":
            alpha = int(255 * self._alpha)
            painter.fillRect(self.rect(), QColor(0, 0, 0, alpha))

        painter.end()