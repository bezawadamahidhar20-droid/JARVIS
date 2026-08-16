"""Holographic Data Panels — the left "AI NEURAL NETWORK" panel and the right
"INTELLIGENCE CORE" dashboard.

Both are drawn on the same ``HoloPanel`` base that provides:
* a semi-transparent dark fill with a blue tint,
* a glowing border that pulses slowly,
* a holographic title bar with the thin underline,
* an optional typewriter effect for text.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient
from PySide6.QtWidgets import QWidget, QVBoxLayout

from jarvis_ui.ui_state import LISTENING, THINKING, SPEAKING
from jarvis_ui.widgets.brain_map import BrainMapWidget

# ── Colours (configurable) ────────────────────────────────────────────────────
PANEL_FILL = (0.01, 0.03, 0.08, 0.72)
PANEL_BORDER = (0.0, 0.6, 1.0, 0.55)
TITLE_COLOR = (0.0, 0.9, 1.0, 1.0)
TEXT_COLOR = (0.75, 0.85, 0.95, 0.9)
TEXT_DIM = (0.35, 0.5, 0.65, 0.8)
ACCENT = (1.0, 0.42, 0.0, 1.0)          # iron-man orange
SUCCESS = (0.0, 1.0, 0.53, 1.0)         # green pulse
WARNING = (1.0, 0.2, 0.2, 1.0)
BAR_BG = (0.0, 0.15, 0.3, 0.6)
BAR_FILL = (0.0, 0.8, 1.0, 0.9)

TYPEWRITER_CHARS_PER_SEC = 48


def _rgba(color, alpha: float | None = None) -> QColor:
    if alpha is None:
        alpha = color[3]
    return QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(alpha * 255))


class TypewriterText:
    """Reveals a string character-by-character for the typewriter effect."""

    def __init__(self) -> None:
        self.full = ""
        self.shown = 0
        self._timer: float = 0.0

    def set_text(self, text: str, reset: bool = True) -> None:
        self.full = text or ""
        if reset:
            self.shown = 0
            self._timer = 0.0

    def tick(self, dt: float) -> None:
        if self.shown < len(self.full):
            self._timer += dt
            step = int(self._timer * TYPEWRITER_CHARS_PER_SEC)
            self.shown = min(len(self.full), self.shown + max(1, step))
            self._timer = 0.0 if step == 0 else self._timer

    @property
    def visible(self) -> str:
        return self.full[: self.shown]


class HoloPanel(QWidget):
    """Base holographic panel with a pulsing glow border + title bar."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self._t = 0.0
        self.setMinimumHeight(120)

    def tick(self, dt: float) -> None:
        self._t += dt
        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 12.0

        # Fill.
        painter.setPen(Qt.NoPen)
        painter.setBrush(_rgba(PANEL_FILL))
        painter.drawRoundedRect(rect, radius, radius)

        # Pulsing border.
        pulse = 0.5 + 0.5 * math.sin(self._t * 2.2)
        border = QColor(*[int(c * 255) for c in PANEL_BORDER[:3]], int((90 + 130 * pulse)))
        pen = QPen(border, 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)

        # Title bar.
        title_rect = QRectF(rect.x() + 14, rect.y() + 10, rect.width() - 28, 20)
        painter.setFont(QFont("Courier New", 9, QFont.Bold))
        painter.setPen(_rgba(TITLE_COLOR))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title)

        # Thin underline under title.
        painter.setPen(_rgba(PANEL_BORDER, 150 + 100 * pulse))
        painter.drawLine(QPointF(rect.x() + 14, rect.y() + 32), QPointF(rect.right() - 14, rect.y() + 32))

        # Corner brackets (holographic HUD flourish).
        bracket = QColor(0, 180, 255, 220)
        painter.setPen(QPen(bracket, 2))
        corner = 8.0
        # top-left
        painter.drawLine(QPointF(rect.x(), rect.y() + corner), QPointF(rect.x(), rect.y()))
        painter.drawLine(QPointF(rect.x(), rect.y()), QPointF(rect.x() + corner, rect.y()))
        # top-right
        painter.drawLine(QPointF(rect.right() - corner, rect.y()), QPointF(rect.right(), rect.y()))
        painter.drawLine(QPointF(rect.right(), rect.y()), QPointF(rect.right(), rect.y() + corner))
        # bottom-left
        painter.drawLine(QPointF(rect.x(), rect.bottom() - corner), QPointF(rect.x(), rect.bottom()))
        painter.drawLine(QPointF(rect.x(), rect.bottom()), QPointF(rect.x() + corner, rect.bottom()))
        # bottom-right
        painter.drawLine(QPointF(rect.right() - corner, rect.bottom()), QPointF(rect.right(), rect.bottom()))
        painter.drawLine(QPointF(rect.right(), rect.bottom()), QPointF(rect.right(), rect.bottom() - corner))

        self._paint_content(painter, rect)
        painter.end()

    def _paint_content(self, painter: QPainter, rect: QRectF) -> None:
        """Override in subclasses to draw panel-specific content."""
        raise NotImplementedError


def _draw_bar(painter: QPainter, x: float, y: float, w: float, h: float, frac: float, color=BAR_FILL) -> None:
    frac = max(0.0, min(1.0, frac))
    painter.setPen(Qt.NoPen)
    painter.setBrush(_rgba(BAR_BG))
    painter.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)
    if frac > 0.01:
        painter.setBrush(_rgba(color))
        painter.drawRoundedRect(QRectF(x, y, w * frac, h), h / 2, h / 2)


# ── Left panel: AI NEURAL NETWORK ─────────────────────────────────────────────

class _LeftStats(QWidget):
    """Processing stats block that lives below the brain map."""

    def __init__(self, state=None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setFixedHeight(84)
        self._t = 0.0

    def tick(self, dt: float) -> None:
        self._t += dt
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        stats = self.state.snapshot() if self.state else {}
        status = stats.get("status", "idle")

        rows = [
            (f"Tokens/sec : {stats.get('tokens_per_sec', 0):6.0f}", TEXT_COLOR),
            ("Model layers: 32", TEXT_COLOR),
            ("Context win : 2048", TEXT_COLOR),
        ]
        y = 4.0
        painter.setFont(QFont("Courier New", 8))
        for text, color in rows:
            painter.setPen(_rgba(color))
            painter.drawText(QRectF(4, y, self.width() - 8, 14), Qt.AlignLeft | Qt.AlignVCenter, text)
            y += 15

        progress = stats.get("thinking_progress", 0.0)
        if status == THINKING:
            progress = 0.35 + 0.6 * abs(math.sin(self._t * 1.5))
        painter.setFont(QFont("Courier New", 7, QFont.Bold))
        label = "THINKING..." if status == THINKING else "STANDBY"
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(4, y, 90, 14), Qt.AlignLeft | Qt.AlignVCenter, label)
        _draw_bar(painter, 100, y + 2, self.width() - 104, 8, progress, ACCENT if status == THINKING else BAR_FILL)
        painter.end()


class LeftPanel(QWidget):
    """Combines the BrainMap + processing stats + thinking progress bar."""

    def __init__(self, state=None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._t = 0.0
        self.title = "AI NEURAL NETWORK"
        self.setMinimumWidth(330)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 38, 12, 10)
        self._layout.setSpacing(8)

        self._map = BrainMapWidget(state)
        self._layout.addWidget(self._map, 1)

        self._stats = _LeftStats(state)
        self._layout.addWidget(self._stats)

    def tick(self, dt: float) -> None:
        self._t += dt
        activity = self.state.get_module_activity() if self.state else {}
        self._map.update_frame(dt, activity)
        self._stats.tick(dt)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 12.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(_rgba(PANEL_FILL))
        painter.drawRoundedRect(rect, radius, radius)

        pulse = 0.5 + 0.5 * math.sin(self._t * 2.2)
        border = QColor(*[int(c * 255) for c in PANEL_BORDER[:3]], int((90 + 130 * pulse)))
        painter.setPen(QPen(border, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)

        painter.setFont(QFont("Courier New", 9, QFont.Bold))
        painter.setPen(_rgba(TITLE_COLOR))
        painter.drawText(QRectF(rect.x() + 14, rect.y() + 8, rect.width() - 28, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        painter.setPen(_rgba(PANEL_BORDER, 150 + 100 * pulse))
        painter.drawLine(QPointF(rect.x() + 14, rect.y() + 30), QPointF(rect.right() - 14, rect.y() + 30))
        painter.end()


# ── Right panel: INTELLIGENCE CORE ────────────────────────────────────────────

class RightPanel(QWidget):
    """Holds three HoloPanels: conversation, processing status, system stats."""

    def __init__(self, state=None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._t = 0.0
        self.setMinimumWidth(360)
        self._typewriter = TypewriterText()
        self._scroll_offset = 0.0
        self._conv_cache: list[dict] = []

    def tick(self, dt: float) -> None:
        self._t += dt
        conv = self.state.get_conversation() if self.state else []
        if conv and conv[-1].get("text") != self._typewriter.full:
            self._typewriter.set_text(conv[-1]["text"])
        self._typewriter.tick(dt)
        self._conv_cache = conv
        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        pulse = 0.5 + 0.5 * math.sin(self._t * 2.2)
        border_c = QColor(*[int(c * 255) for c in PANEL_BORDER[:3]], int((90 + 130 * pulse)))

        panel_w = w - 20
        gap = 10
        y = 6

        def panel(x: float, y: float, pw: float, ph: float, title: str) -> None:
            rect = QRectF(x, y, pw, ph)
            radius = 10.0
            painter.setPen(Qt.NoPen)
            painter.setBrush(_rgba(PANEL_FILL))
            painter.drawRoundedRect(rect, radius, radius)
            painter.setPen(QPen(border_c, 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)
            painter.setFont(QFont("Courier New", 8, QFont.Bold))
            painter.setPen(_rgba(TITLE_COLOR))
            painter.drawText(QRectF(rect.x() + 10, rect.y() + 6, rect.width() - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, title)
            painter.setPen(_rgba(PANEL_BORDER, 120 + 100 * pulse))
            painter.drawLine(QPointF(rect.x() + 10, rect.y() + 24), QPointF(rect.right() - 10, rect.y() + 24))

        # Panel 1: Conversation history (last 5).
        p1_h = h * 0.30
        panel(10, y, panel_w, p1_h, "CONVERSATION")
        self._draw_conversation(painter, 20, y + 28, panel_w - 20, p1_h - 34)
        y += p1_h + gap

        # Panel 2: Processing status.
        p2_h = h * 0.34
        panel(10, y, panel_w, p2_h, "PROCESSING STATUS")
        self._draw_processing(painter, 20, y + 30, panel_w - 20, p2_h - 36)
        y += p2_h + gap

        # Panel 3: System stats.
        p3_h = h - y - 10
        if p3_h < 80:
            p3_h = 80
        panel(10, y, panel_w, p3_h, "SYSTEM STATUS")
        self._draw_system(painter, 20, y + 30, panel_w - 20, p3_h - 36)

        painter.end()

    def _draw_conversation(self, painter: QPainter, x: float, y: float, w: float, h: float) -> None:
        conv = self._conv_cache[-5:]
        if not conv:
            painter.setFont(QFont("Courier New", 8))
            painter.setPen(_rgba(TEXT_DIM))
            painter.drawText(QRectF(x, y, w, h), Qt.AlignCenter, "NO CONVERSATION YET")
            return

        painter.setFont(QFont("Courier New", 7))
        row_h = 34.0
        vis_rows = max(1, int(h // row_h))
        start = max(0, len(conv) - vis_rows)
        yy = y
        for entry in conv[start:]:
            role = entry.get("role", "user")
            text = entry.get("text", "")
            if len(text) > 46:
                text = text[:45] + "..."
            label = "YOU  " if role == "user" else "JARVIS"
            color = TEXT_DIM if role == "user" else TITLE_COLOR
            painter.setPen(_rgba(color))
            painter.setFont(QFont("Courier New", 7, QFont.Bold))
            painter.drawText(QRectF(x, yy, 60, 12), Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.setPen(_rgba(TEXT_COLOR))
            painter.setFont(QFont("Courier New", 7))
            painter.drawText(QRectF(x + 62, yy, w - 62, row_h - 6), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)
            yy += row_h
            if yy > y + h - row_h:
                break

    def _draw_processing(self, painter: QPainter, x: float, y: float, w: float, h: float) -> None:
        stats = self.state.snapshot() if self.state else {}
        status = stats.get("status", "idle")
        model = "Qwen3 8B"
        painter.setFont(QFont("Courier New", 8))
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Model")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, model)
        y += 16

        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Status")
        status_color = {
            LISTENING: _rgba(WARNING),
            THINKING: _rgba(TITLE_COLOR),
            SPEAKING: _rgba(ACCENT),
        }.get(status, _rgba(TEXT_DIM))
        painter.setPen(status_color)
        painter.setFont(QFont("Courier New", 8, QFont.Bold))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, status.upper())
        y += 16

        painter.setFont(QFont("Courier New", 8))
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Response time")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{stats.get('response_time', 0.0):.1f}s")
        y += 16

        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Memory turns")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{stats.get('memory_turns', 0)}/20")
        y += 20

        # Intent classification.
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Intent")
        painter.setPen(_rgba(TITLE_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, stats.get("current_intent", "AI_QUESTION"))
        y += 16
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Confidence")
        conf = stats.get("confidence", 0.0)
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{conf * 100:.0f}%")
        _draw_bar(painter, x + 130, y + 2, w - 130, 6, conf, BAR_FILL)
        y += 18
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Route")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, stats.get("route", "Qwen3 Brain"))
        y += 22

        # Response quality meter (animated).
        q = 0.55 + 0.35 * abs(math.sin(self._t * 0.9))
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 130, 14), Qt.AlignLeft | Qt.AlignVCenter, "Quality")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 130, y, w - 130, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{q * 100:.0f}%")
        _draw_bar(painter, x + 130, y + 2, w - 130, 6, q, SUCCESS)
        y += 20

        # Current response (typewriter).
        if self._typewriter.visible:
            painter.setPen(_rgba(TITLE_COLOR))
            painter.setFont(QFont("Courier New", 7))
            text_rect = QRectF(x, y, w, max(10.0, h - (y - self.rect().y() - 40)))
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, self._typewriter.visible)

    def _draw_system(self, painter: QPainter, x: float, y: float, w: float, h: float) -> None:
        stats = self.state.snapshot() if self.state else {}
        cpu = stats.get("cpu_usage", 0.0) / 100.0
        ram = stats.get("ram_usage", 0.0) / 100.0

        painter.setFont(QFont("Courier New", 8))
        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 100, 14), Qt.AlignLeft | Qt.AlignVCenter, "CPU")
        painter.setPen(_rgba(TEXT_COLOR))
        painter.drawText(QRectF(x + 100, y, w - 100, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{cpu * 100:.0f}%")
        _draw_bar(painter, x, y + 15, w, 6, cpu, BAR_FILL if cpu < 0.7 else WARNING)
        y += 28

        painter.setPen(_rgba(TEXT_DIM))
        painter.drawText(QRectF(x, y, 100, 14), Qt.AlignLeft | Qt.AlignVCenter, "RAM")
        painter.setPen(_rgba(TEXT_COLOR))
        ram_gb = ram * 32.0
        painter.drawText(QRectF(x + 100, y, w - 100, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{ram_gb:.1f} GB")
        _draw_bar(painter, x, y + 15, w, 6, ram, BAR_FILL if ram < 0.7 else WARNING)
        y += 32

        rows = [
            ("OLLAMA", stats.get("ollama_connected", False)),
            ("MIC", stats.get("mic_available", False)),
            ("TTS", stats.get("tts_available", True)),
        ]
        for label, ok in rows:
            color = SUCCESS if ok else WARNING
            painter.setPen(_rgba(TEXT_DIM))
            painter.drawText(QRectF(x, y, 60, 14), Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.setPen(_rgba(color))
            status_text = "CONNECTED" if ok else "OFFLINE"
            painter.drawText(QRectF(x + 60, y, w - 60, 14), Qt.AlignLeft | Qt.AlignVCenter, status_text)
            y += 18