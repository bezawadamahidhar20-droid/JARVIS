"""Background Particle System.

A full-window layer of floating dots animated entirely in numpy.

Array layout:  particles of shape ``(N, 6)`` → columns are
``x, y, vx, vy, life, brightness``.  All positions update in one vectorised
numpy operation per frame for performance at 60fps.

Behaviour
---------
* IDLE      — slow random drift, gentle respawn.
* LISTENING — subtle inward pull toward the center sphere.
* THINKING  — particles accelerate and converge toward the center.
* SPEAKING  — particles explode outward from the sphere.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget

from jarvis_ui.ui_state import LISTENING, THINKING, SPEAKING

# ── Colours (configurable) ────────────────────────────────────────────────────
COLOR_DARK = (0.02, 0.06, 0.15)          # mostly dark blue
COLOR_BRIGHT = (0.0, 0.9, 1.0)           # occasional bright cyan flash

N_PARTICLES = 420
BASE_SPEED = 18.0
CONVERGE_SPEED = 95.0
EXPLODE_SPEED = 160.0


class ParticleLayer(QWidget):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.state = state
        self._particles = np.zeros((N_PARTICLES, 6), dtype=np.float64)
        self._t = 0.0
        self._status = "idle"
        self._init_particles()

    def _init_particles(self) -> None:
        rng = np.random.default_rng(7)
        n = N_PARTICLES
        w = max(1, self.width())
        h = max(1, self.height())
        self._particles[:, 0] = rng.uniform(0, w, n)   # x
        self._particles[:, 1] = rng.uniform(0, h, n)   # y
        self._particles[:, 2] = rng.uniform(-BASE_SPEED, BASE_SPEED, n)  # vx
        self._particles[:, 3] = rng.uniform(-BASE_SPEED, BASE_SPEED, n)  # vy
        self._particles[:, 4] = rng.uniform(0.3, 1.0, n)  # life
        self._particles[:, 5] = rng.uniform(0.1, 0.6, n)  # brightness

    # ── Simulation ─────────────────────────────────────────────────────────

    def update_frame(self, dt: float, status: str, w: int, h: int) -> None:
        self._t += dt
        self._status = status

        if w <= 0 or h <= 0:
            return
        if self._particles.shape[0] == 0:
            self._init_particles()

        p = self._particles
        cx, cy = w / 2.0, h / 2.0
        dx = cx - p[:, 0]
        dy = cy - p[:, 1]
        dist = np.hypot(dx, dy) + 1e-6

        # Base drift.
        p[:, 2] += np.random.uniform(-30, 30, p.shape[0]) * dt
        p[:, 3] += np.random.uniform(-30, 30, p.shape[0]) * dt

        # Status-specific behaviour.
        if status == THINKING:
            pull = CONVERGE_SPEED
            p[:, 2] += (dx / dist) * pull * dt
            p[:, 3] += (dy / dist) * pull * dt
        elif status == LISTENING:
            p[:, 2] += (dx / dist) * 22 * dt
            p[:, 3] += (dy / dist) * 22 * dt
        elif status == SPEAKING:
            # Explode outward from center.
            p[:, 2] += (dx / dist) * EXPLODE_SPEED * dt
            p[:, 3] += (dy / dist) * EXPLODE_SPEED * dt

        # Integrate.
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt

        # Life decay → brightness flicker.
        p[:, 4] -= dt * (0.25 if status == THINKING else 0.08)
        p[:, 5] = np.clip(p[:, 5] + np.random.uniform(-0.3, 0.3, p.shape[0]) * dt, 0.05, 1.0)

        # Respawn dead / out-of-bounds particles.
        dead = (p[:, 4] <= 0.0) | (p[:, 0] < -40) | (p[:, 0] > w + 40) | (p[:, 1] < -40) | (p[:, 1] > h + 40)
        n_dead = int(dead.sum())
        if n_dead:
            p[dead, 0] = np.random.uniform(-20, w + 20, n_dead)
            p[dead, 1] = np.random.uniform(-20, h + 20, n_dead)
            p[dead, 2] = np.random.uniform(-BASE_SPEED, BASE_SPEED, n_dead)
            p[dead, 3] = np.random.uniform(-BASE_SPEED, BASE_SPEED, n_dead)
            p[dead, 4] = np.random.uniform(0.5, 1.0, n_dead)
            if status == SPEAKING:
                # Spawn freshly-exploded particles at the center.
                p[dead, 0] = cx
                p[dead, 1] = cy
                p[dead, 2] = np.random.uniform(-EXPLODE_SPEED, EXPLODE_SPEED, n_dead)
                p[dead, 3] = np.random.uniform(-EXPLODE_SPEED, EXPLODE_SPEED, n_dead)

        self.update()

    # ── Painting ───────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)

        if self._particles.shape[0] == 0:
            painter.end()
            return

        p = self._particles
        bright = p[:, 5]
        # Brightness → occasional cyan flash.
        for i in range(0, p.shape[0], 2):
            x, y = p[i, 0], p[i, 1]
            if x < 0 or y < 0 or x > self.width() or y > self.height():
                continue
            b = float(bright[i])
            alpha = int(30 + 140 * b)
            if b > 0.85:
                r, g, bl = COLOR_BRIGHT
            else:
                r, g, bl = COLOR_DARK
            size = 1.0 + 1.5 * b
            painter.setBrush(QColor(int(r * 255), int(g * 255), int(bl * 255), alpha))
            painter.drawRect(int(x), int(y), int(size), int(size))

        painter.end()