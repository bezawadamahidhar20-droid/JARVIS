"""AI Brain Topology Map — the workflow visualizer.

Shows the JARVIS module pipeline as a 3D-looking graph of hexagonal nodes:

    MICROPHONE -> STT -> ROUTER -> OLLAMA/QWEN3
                             \\-> COMMANDS
    OLLAMA -> MEMORY -> TTS -> SPEAKER

When a module is *active* its node lights up bright cyan, connecting lines
pulse and animated particles travel along them.  Active modules come from
``state.module_activity`` (fed by the backend thread).

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt, QRectF, QPointF, QLineF
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
from PySide6.QtWidgets import QWidget

# ── Colours (configurable) ────────────────────────────────────────────────────
BG_FILL = (0.01, 0.03, 0.08, 0.0)
NODE_IDLE = (0.02, 0.25, 0.5, 0.85)       # dim blue hexagon
NODE_ACTIVE = (0.0, 0.9, 1.0, 0.95)       # bright cyan when processing
NODE_EDGE = (0.0, 0.6, 1.0, 0.8)
TEXT_IDLE = (0.55, 0.75, 0.95, 0.9)
TEXT_ACTIVE = (1.0, 1.0, 1.0, 1.0)
EDGE_IDLE = (0.0, 0.3, 0.6, 0.35)
EDGE_ACTIVE = (0.0, 0.9, 1.0, 0.9)
PARTICLE_COLOR = (1.0, 0.9, 0.6, 1.0)     # traveling energy particles

# ── Layout (normalised 0..1 within the widget) ────────────────────────────────
NODE_DEFS = {
    "MICROPHONE": (0.14, 0.70),
    "STT": (0.40, 0.70),
    "ROUTER": (0.62, 0.70),
    "OLLAMA": (0.86, 0.38),
    "COMMANDS": (0.62, 0.20),
    "MEMORY": (0.86, 0.55),
    "TTS": (0.86, 0.78),
    "SPEAKER": (0.62, 0.95),
}

# Directed connections (source -> target).  Keys match NODE_DEFS.
CONNECTIONS = [
    ("MICROPHONE", "STT"),
    ("STT", "ROUTER"),
    ("ROUTER", "OLLAMA"),
    ("ROUTER", "COMMANDS"),
    ("OLLAMA", "MEMORY"),
    ("MEMORY", "TTS"),
    ("TTS", "SPEAKER"),
]

HEX_RX = 0.115  # horizontal hexagon half-size (fraction of width)
HEX_RY = 0.055  # vertical half-size


class BrainMapWidget(QWidget):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.state = state
        self.setMinimumHeight(260)
        self._t = 0.0
        self._particles: list[dict] = []

    def update_frame(self, dt: float, activity: dict[str, float]) -> None:
        self._t += dt
        activity = activity or {}
        # Spawn traveling particles on active edges.
        active_src = [src for src, _ in CONNECTIONS if activity.get(src, 0.0) > 0.5]
        if active_src and random.random() < 8.0 * dt:
            src = random.choice(active_src)
            targets = [t for s, t in CONNECTIONS if s == src]
            if targets:
                self._particles.append({"src": src, "dst": random.choice(targets), "t": 0.0})
        # Advance particles.
        kept = []
        for p in self._particles:
            p["t"] += dt * 0.9
            if p["t"] < 1.0:
                kept.append(p)
        self._particles = kept
        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

        w, h = self.width(), self.height()
        activity = self.state.get_module_activity() if self.state else {}

        def pos(name: str) -> QPointF:
            nx, ny = NODE_DEFS[name]
            return QPointF(nx * w, ny * h)

        # ── Connections ────────────────────────────────────────────────────
        for src, dst in CONNECTIONS:
            p1, p2 = pos(src), pos(dst)
            active = activity.get(src, 0.0) > 0.5
            color = EDGE_ACTIVE if active else EDGE_IDLE
            alpha = (color[3] * 255) * (0.6 + 0.4 * math.sin(self._t * 4)) if active else color[3] * 255
            pen = QPen(QColor(*[int(c * 255) for c in color[:3]], int(alpha)))
            pen.setWidth(2 if active else 1)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(p1, p2)

        # ── Traveling particles ────────────────────────────────────────────
        for p in self._particles:
            p1, p2 = pos(p["src"]), pos(p["dst"])
            t = p["t"]
            x = p1.x() + (p2.x() - p1.x()) * t
            y = p1.y() + (p2.y() - p1.y()) * t
            alpha = int(255 * (0.5 + 0.5 * math.sin(math.pi * t)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(*[int(c * 255) for c in PARTICLE_COLOR], alpha))
            r = 2.5 + 2.5 * math.sin(math.pi * t)
            painter.drawEllipse(QPointF(x, y), r, r)

        # ── Nodes ──────────────────────────────────────────────────────────
        for name, (_, _) in NODE_DEFS.items():
            center = pos(name)
            is_active = activity.get(name, 0.0) > 0.5
            pulse = 0.5 + 0.5 * math.sin(self._t * 5 + NODE_DEFS[name][0] * 20)
            rx = HEX_RX * w
            ry = HEX_RY * h
            if is_active:
                rx *= 1.0 + 0.12 * pulse
                ry *= 1.0 + 0.12 * pulse

            # Hexagon outline + fill.
            hexagon = QPainterPath()
            for k in range(6):
                ang = math.pi / 3 * k - math.pi / 2
                px = center.x() + rx * math.cos(ang)
                py = center.y() + ry * math.sin(ang)
                if k == 0:
                    hexagon.moveTo(px, py)
                else:
                    hexagon.lineTo(px, py)
            hexagon.closeSubpath()

            fill = NODE_ACTIVE if is_active else NODE_IDLE
            fill_alpha = int(fill[3] * 255 * (0.7 + 0.3 * pulse if is_active else 1.0))
            painter.setBrush(QColor(*[int(c * 255) for c in fill[:3]], fill_alpha))
            pen = QPen(QColor(*[int(c * 255) for c in NODE_EDGE[:3]], int(120 + 135 * pulse) if is_active else 160))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawPath(hexagon)

            # Glow ring when active.
            if is_active:
                pen = QPen(QColor(*[int(c * 255) for c in NODE_ACTIVE[:3]], int(90 + 90 * pulse)))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, rx * 1.35, ry * 1.35)

            # Label.
            label = name.replace("_", " ")
            painter.setFont(QFont("Courier New", 7, QFont.Bold))
            color = TEXT_ACTIVE if is_active else TEXT_IDLE
            painter.setPen(QColor(*[int(c * 255) for c in color[:3]], int(color[3] * 255)))
            painter.drawText(center.x() - rx, center.y() - ry - 8, rx * 2, ry * 2, Qt.AlignCenter, label)

        painter.end()