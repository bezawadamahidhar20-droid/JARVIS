"""Top status bar + bottom control bar.

Top bar
-------
* Left   — "J.A.R.V.I.S" glowing neon title.
* Center — animated status icon + status text.
* Right  — live clock + date.
* Bottom — thin scanning line that sweeps across.

Bottom bar
----------
* Circular MIC button (idle: blue ring, listening: pulsing red ring).
* Text input field (left of mic) — Enter sends a typed command.
* Quick command buttons: Time / Date / YouTube / Clear Memory.
* Status indicator dots (Ollama / Mic / TTS).

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math
import time
from datetime import datetime

from PySide6.QtCore import Qt, QRectF, QPointF, QSignalMapper, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QFontMetricsF
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
)

from jarvis_ui.ui_state import LISTENING, THINKING, SPEAKING, IDLE, SHUTDOWN

# ── Colours (configurable) ────────────────────────────────────────────────────
TITLE_GLOW = (0.0, 0.8, 1.0, 1.0)
TITLE_CORE = (1.0, 1.0, 1.0, 1.0)
BAR_BG = (0.0, 0.0, 0.0, 0.85)
SCAN_COLOR = (0.0, 0.9, 1.0, 0.55)
MIC_IDLE = (0.0, 0.6, 1.0, 0.9)
MIC_ACTIVE = (1.0, 0.2, 0.2, 1.0)
MIC_CORE = (0.8, 0.85, 1.0, 1.0)
DOT_OK = (0.0, 1.0, 0.53, 1.0)
DOT_OFF = (1.0, 0.2, 0.2, 1.0)
BTN_BG = (0.02, 0.1, 0.2, 0.8)
BTN_TEXT = (0.7, 0.9, 1.0, 0.95)
INPUT_BG = (0.01, 0.05, 0.12, 0.9)
INPUT_TEXT = (0.85, 0.92, 1.0, 1.0)


class TopStatusBar(QWidget):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.state = state
        self.setFixedHeight(64)
        self._t = 0.0
        self._time_now = datetime.now()

    def tick(self, dt: float) -> None:
        self._t += dt
        self._time_now = datetime.now()
        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(*[int(c * 255) for c in BAR_BG]))
        # Bottom border line.
        painter.setPen(QPen(QColor(0, 60, 130, 120), 1))
        painter.drawLine(QPointF(0, h - 2), QPointF(w, h - 2))

        stats = self.state.snapshot() if self.state else {}
        status = stats.get("status", IDLE)

        # ── Left: J.A.R.V.I.S title ───────────────────────────────────────
        painter.setFont(QFont("Courier New", 18, QFont.Bold))
        title = "J.A.R.V.I.S"
        tw = painter.fontMetrics().horizontalAdvance(title)
        # Glow pass.
        painter.setPen(QPen(QColor(*[int(c * 255) for c in TITLE_GLOW[:3]], 90), 1))
        painter.drawText(QRectF(18, 8, w, h - 10), Qt.AlignLeft | Qt.AlignVCenter, title)
        # Core pass.
        painter.setPen(QColor(*[int(c * 255) for c in TITLE_CORE]))
        painter.drawText(QRectF(18, 8, w, h - 10), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.setFont(QFont("Courier New", 8))
        painter.setPen(QColor(90, 140, 200, 200))
        painter.drawText(QRectF(20 + tw + 8, 8, w, h - 10), Qt.AlignLeft | Qt.AlignVCenter, "v2.0  //  OLLAMA 0.6.0")

        # ── Center: status icon + text ────────────────────────────────────
        cx = w / 2
        icon_x = cx - 130
        label_x = cx - 105
        icon_r = 7.0
        pulse = 0.5 + 0.5 * math.sin(self._t * 6.0)

        status_color = {
            LISTENING: (255, 60, 60),
            THINKING: (0, 220, 255),
            SPEAKING: (255, 160, 60),
            IDLE: (120, 160, 220),
            SHUTDOWN: (255, 60, 60),
        }.get(status, (120, 160, 220))
        label_text = {
            LISTENING: "LISTENING...",
            THINKING: "THINKING...",
            SPEAKING: "SPEAKING...",
            IDLE: "IDLE",
            SHUTDOWN: "SHUTDOWN",
        }.get(status, "IDLE")

        # Animated icon: expanding ring for active states.
        if status in (LISTENING, THINKING, SPEAKING):
            painter.setPen(QPen(QColor(*status_color, int(60 + 140 * pulse)), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(icon_x, h / 2), icon_r + 5 + 4 * pulse, icon_r + 5 + 4 * pulse)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*status_color, 255))
        painter.drawEllipse(QPointF(icon_x, h / 2), icon_r, icon_r)

        painter.setFont(QFont("Courier New", 12, QFont.Bold))
        painter.setPen(QColor(*status_color, 255))
        painter.drawText(QRectF(label_x, 8, 260, h - 10), Qt.AlignLeft | Qt.AlignVCenter, label_text)

        # ── Right: clock + date ───────────────────────────────────────────
        painter.setFont(QFont("Courier New", 12, QFont.Bold))
        painter.setPen(QColor(200, 230, 255, 255))
        time_text = self._time_now.strftime("%H:%M:%S")
        date_text = self._time_now.strftime("%A, %d %B %Y")
        painter.drawText(QRectF(w - 250, 6, 232, 26), Qt.AlignRight | Qt.AlignVCenter, time_text)
        painter.setFont(QFont("Courier New", 8))
        painter.setPen(QColor(120, 160, 210, 220))
        painter.drawText(QRectF(w - 250, 34, 232, 18), Qt.AlignRight | Qt.AlignVCenter, date_text)

        # ── Scanning line at the very bottom ──────────────────────────────
        scan_x = (self._t * 120) % (w + 200) - 100
        pen = QPen(QColor(*[int(c * 255) for c in SCAN_COLOR[:3]], 160))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(QPointF(scan_x, h - 1), QPointF(scan_x + 90, h - 1))

        painter.end()


# ── Bottom control bar ────────────────────────────────────────────────────────

class MicButton(QWidget):
    """Circular push-to-talk button.  Signal ``toggled(bool)`` when clicked."""

    toggled = Signal(bool)

    def __init__(self, state=None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setFixedSize(64, 64)
        self._active = False
        self._t = 0.0
        self._pressed = False

    def tick(self, dt: float) -> None:
        self._t += dt
        self.update()

    def set_active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._pressed = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._pressed and self.rect().contains(event.position().toPoint()):
            self._active = not self._active
            self.toggled.emit(self._active)
        self._pressed = False
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = 32, 32
        r = 26.0
        pulse = 0.5 + 0.5 * math.sin(self._t * 6.0)

        # Outer glow ring.
        if self._active:
            color = MIC_ACTIVE
            for i in range(3):
                alpha = int(120 * (1 - i / 3) * (0.5 + 0.5 * pulse))
                painter.setPen(QPen(QColor(*[int(c * 255) for c in MIC_ACTIVE[:3]], alpha), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), r + 4 + i * 6 + 4 * pulse, r + 4 + i * 6 + 4 * pulse)
        else:
            color = MIC_IDLE
            painter.setPen(QPen(QColor(*[int(c * 255) for c in MIC_IDLE[:3]], 110), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r + 4 + 3 * pulse, r + 4 + 3 * pulse)

        # Button body.
        painter.setPen(QPen(QColor(*[int(c * 255) for c in color[:3]], 255), 2))
        painter.setBrush(QColor(6, 14, 28, 200))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Mic glyph (drawn as simple shapes).
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*[int(c * 255) for c in MIC_CORE[:3]], 235))
        painter.drawRoundedRect(QRectF(cx - 5, cy - 12, 10, 14), 3, 3)  # capsule
        painter.drawRect(int(cx - 7), int(cy + 3), 14, 2)  # capsule bottom cap
        painter.drawRect(int(cx - 1), int(cy + 3), 2, 6)   # stem
        painter.drawRoundedRect(QRectF(cx - 8, cy + 9, 16, 2), 1, 1)  # base

        painter.end()


class StatusDot(QWidget):
    """Small green/red indicator with label."""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.label = label
        self.ok = False
        self.setFixedHeight(20)

    def set_ok(self, ok: bool) -> None:
        self.ok = ok
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = DOT_OK if self.ok else DOT_OFF
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*[int(c * 255) for c in color[:3]], 255))
        painter.drawEllipse(QPointF(8, 10), 4, 4)
        painter.setPen(QColor(160, 200, 240, 220))
        painter.setFont(QFont("Courier New", 7, QFont.Bold))
        painter.drawText(QRectF(16, 0, 70, 20), Qt.AlignLeft | Qt.AlignVCenter, self.label)
        painter.end()


class BottomBar(QWidget):
    """Bottom control bar: mic button + text input + quick command buttons."""

    def __init__(self, state=None, on_send=None, on_mic_toggle=None, on_quick=None) -> None:
        super().__init__()
        self.state = state
        self.on_send = on_send
        self.on_mic_toggle = on_mic_toggle
        self.on_quick = on_quick
        self.setFixedHeight(90)
        self._t = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        # Status dots.
        dots = QWidget(self)
        dots_layout = QVBoxLayout(dots)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(4)
        self.dot_ollama = StatusDot("OLLAMA")
        self.dot_mic = StatusDot("MIC")
        self.dot_tts = StatusDot("TTS")
        for d in (self.dot_ollama, self.dot_mic, self.dot_tts):
            dots_layout.addWidget(d)
        layout.addWidget(dots)

        # Text input.
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("Type a question or command, press Enter...")
        self.input.setFixedHeight(34)
        self.input.setStyleSheet(
            f"QLineEdit {{ background-color: rgba(3,12,28,230); color: rgb(220,240,255);"
            f" border: 1px solid rgb(0,90,180); border-radius: 6px; padding: 0 10px;"
            f" font-family: 'Courier New'; font-size: 12px; }}"
            f"QLineEdit:focus {{ border: 1px solid rgb(0,200,255); }}"
        )
        self.input.returnPressed.connect(self._send)
        layout.addWidget(self.input, 1)

        # Mic button.
        self.mic = MicButton(state, self)
        self.mic.toggled.connect(self._on_mic_toggled)
        layout.addWidget(self.mic)

        # Quick command buttons.
        for label in ("TIME", "DATE", "YOUTUBE", "CLEAR MEMORY"):
            btn = QPushButton(label, self)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgba(5,25,50,220); color: rgb(180,225,255);"
                f" border: 1px solid rgb(0,110,200); border-radius: 4px;"
                f" font-family: 'Courier New'; font-size: 9px; font-weight: bold; }}"
                f"QPushButton:hover {{ background-color: rgba(0,80,160,220); color: white; }}"
                f"QPushButton:pressed {{ background-color: rgba(0,180,255,160); }}"
            )
            btn.clicked.connect(lambda _=False, t=label: self._quick(t))
            layout.addWidget(btn)

        # Stretch to keep buttons right-aligned.
        layout.addStretch(0)

    # ── Events ─────────────────────────────────────────────────────────────

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        if self.on_send:
            self.on_send(text)

    def _on_mic_toggled(self, active: bool) -> None:
        if self.state is not None:
            self.state.mic_enabled = active
        if self.on_mic_toggle:
            self.on_mic_toggle(active)

    def _quick(self, label: str) -> None:
        commands = {
            "TIME": "what time is it",
            "DATE": "what's the date",
            "YOUTUBE": "open youtube",
            "CLEAR MEMORY": "clear memory",
        }
        if self.on_quick:
            self.on_quick(commands[label])

    def tick(self, dt: float) -> None:
        self._t += dt
        self.mic.tick(dt)
        if self.state is not None:
            snap = self.state.snapshot()
            self.dot_ollama.set_ok(snap.get("ollama_connected", False))
            self.dot_mic.set_ok(snap.get("mic_available", False))
            self.dot_tts.set_ok(snap.get("tts_available", True))
        self.update()