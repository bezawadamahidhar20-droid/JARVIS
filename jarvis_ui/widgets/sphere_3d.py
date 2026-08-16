"""Central AI Neural Sphere — OpenGL holographic visualization.

Renders, in a single ``QOpenGLWidget``:

* a 3D neural sphere (glowing nodes + interconnecting edges) that rotates,
* node "firing" sparks that travel along edges when JARVIS is thinking,
* three orbiting rings (horizontal / vertical / diagonal) with light trails,
* a scrolling holographic grid floor beneath the sphere.

State machine (driven by :class:`jarvis_ui.ui_state.JARVISState`):

* IDLE     — slow rotation, occasional random node flashes.
* LISTENING— the sphere "breathes" (scales up/down).
* THINKING — nodes fire with traveling sparks, rotation speeds up.
* SPEAKING — the sphere oscillates up/down with the voice level.

All colours are configurable at the top of this file.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

# ── Colours (configurable) ────────────────────────────────────────────────────
BACKGROUND = (0.0, 0.0, 0.0, 1.0)          # pure black
SPHERE_GRID = (0.02, 0.25, 0.55, 0.16)     # faint hologram wireframe
NODE_COLOR = (0.0, 0.85, 1.0, 1.0)         # electric cyan
NODE_CORE = (1.0, 1.0, 1.0, 1.0)           # white-hot centre of a node
EDGE_COLOR = (0.0, 0.55, 1.0, 0.35)        # connecting lines
FIRE_COLOR = (1.0, 0.42, 0.0, 1.0)         # iron-man orange sparks
RING1_COLOR = (0.0, 0.9, 1.0, 0.9)         # horizontal ring (fast)
RING2_COLOR = (0.0, 0.5, 1.0, 0.9)         # vertical ring (medium)
RING3_COLOR = (0.4, 0.8, 1.0, 0.9)         # diagonal ring (slow)
GRID_COLOR = (0.0, 0.25, 0.6, 1.0)         # floor grid lines
GRID_FAR_COLOR = (0.0, 0.1, 0.3, 1.0)      # faded grid at distance
LABEL_COLOR = (0.5, 0.9, 1.0, 0.95)        # ring text
STATUS_COLOR = (0.0, 0.9, 1.0, 1.0)

# ── Geometry parameters ───────────────────────────────────────────────────────
SPHERE_RADIUS = 1.15
LAT_STEPS = 18
LON_STEPS = 36
NODE_COUNT = 110
NEAREST_NEIGHBOURS = 3
FIRE_SPEED = 0.8
CHAIN_PROBABILITY = 0.6

CAMERA_EYE = np.array([0.0, 1.1, 5.2])
CAMERA_TARGET = np.array([0.0, 0.0, 0.0])
CAMERA_UP = np.array([0.0, 1.0, 0.0])

GRID_Y = -2.35
GRID_EXTENT = 12.0
GRID_SPACING = 1.0

RING_PARAMS = [
    # (radius, tilt_deg, orbit_speed, phase, colour)
    (1.85, 30.0, 1.6, 0.0, RING1_COLOR),
    (2.05, 90.0, 0.9, 2.0, RING2_COLOR),
    (2.25, 60.0, 0.45, 4.0, RING3_COLOR),
]
RING_LABELS = ["QWEN3 8B", "MEMORY ACTIVE", "VOICE READY"]
RING_SEGMENTS = 72


def _build_sphere(radius: float, lat: int, lon: int) -> np.ndarray:
    """Return ``(lat+1)*lon`` vertices of a UV sphere."""
    verts = []
    for i in range(lat + 1):
        theta = math.pi * i / lat
        for j in range(lon):
            phi = 2 * math.pi * j / lon
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.cos(theta)
            z = radius * math.sin(theta) * math.sin(phi)
            verts.append((x, y, z))
    return np.asarray(verts, dtype=np.float64)


def _build_edges(nodes: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Connect every node to its k nearest neighbours (no duplicates)."""
    dist = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    edges: set[tuple[int, int]] = set()
    for i in range(len(nodes)):
        order = np.argsort(dist[i])
        for j in order[1 : k + 1]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def _rot_y(pts: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    x = pts[:, 0] * c + pts[:, 2] * s
    z = -pts[:, 0] * s + pts[:, 2] * c
    return np.stack([x, pts[:, 1], z], axis=-1)


def _ring_points(radius: float, tilt_deg: float, angle: float) -> np.ndarray:
    """Circle in the xz-plane, tilted about the X axis by *tilt_deg*."""
    a = np.linspace(0.0, 2.0 * math.pi, RING_SEGMENTS, endpoint=False) + angle
    x = radius * np.cos(a)
    z = radius * np.sin(a)
    tilt = math.radians(tilt_deg)
    y = np.zeros_like(x)
    y2 = y * math.cos(tilt) - z * math.sin(tilt)
    z2 = y * math.sin(tilt) + z * math.cos(tilt)
    return np.stack([x, y2, z2], axis=-1)


class Sphere3DWidget:
    """Pure data + math behind the OpenGL widget.

    Keeping the simulation separate from rendering lets us unit-test the
    animation logic and also powers the QPainter software fallback.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        all_verts = _build_sphere(SPHERE_RADIUS, LAT_STEPS, LON_STEPS)
        idx = rng.choice(len(all_verts), size=NODE_COUNT, replace=False)
        self.nodes = all_verts[idx].astype(np.float64)
        self.edges = _build_edges(self.nodes, NEAREST_NEIGHBOURS)
        self.edges_arr = np.asarray(self.edges, dtype=np.int32)

        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=NODE_COUNT)
        self.node_flash = np.zeros(NODE_COUNT, dtype=np.float64)
        self.fires: list[dict] = []
        self.yaw = 0.0
        self.t = 0.0

        # Wireframe great circles used as the sphere's hologram body.
        self.lat_rings: list[np.ndarray] = []
        for theta_i in range(3, LAT_STEPS, 4):
            pts = []
            for phi in np.linspace(0.0, 2.0 * math.pi, LON_STEPS, endpoint=False):
                pts.append(
                    (
                        SPHERE_RADIUS * math.sin(theta_i * math.pi / LAT_STEPS) * math.cos(phi),
                        SPHERE_RADIUS * math.cos(theta_i * math.pi / LAT_STEPS),
                        SPHERE_RADIUS * math.sin(theta_i * math.pi / LAT_STEPS) * math.sin(phi),
                    )
                )
            self.lat_rings.append(np.asarray(pts, dtype=np.float64))
        self.lon_rings: list[np.ndarray] = []
        for phi_i in range(0, LON_STEPS, 6):
            pts = []
            for theta in np.linspace(0.0, math.pi, LAT_STEPS + 1):
                pts.append(
                    (
                        SPHERE_RADIUS * math.sin(theta) * math.cos(phi_i * 2.0 * math.pi / LON_STEPS),
                        SPHERE_RADIUS * math.cos(theta),
                        SPHERE_RADIUS * math.sin(theta) * math.sin(phi_i * 2.0 * math.pi / LON_STEPS),
                    )
                )
            self.lon_rings.append(np.asarray(pts, dtype=np.float64))

    # ── Simulation update ──────────────────────────────────────────────────

    def update(self, dt: float, status: str, audio_level: float) -> None:
        self.t += dt
        self.yaw += self._spin_speed(status) * dt

        # Decay node flashes.
        self.node_flash *= max(0.0, 1.0 - 3.2 * dt)

        # Advance firing sparks and chain reactions.
        surviving = []
        for fire in self.fires:
            fire["t"] += FIRE_SPEED * dt
            if fire["t"] >= 1.0:
                j = fire["j"]
                self.node_flash[j] = 1.0
                if random.random() < CHAIN_PROBABILITY and status != "idle":
                    nbr = random.choice(self._neighbours_of(j))
                    self.fires.append({"i": j, "j": nbr, "t": 0.0})
            else:
                surviving.append(fire)
        self.fires = surviving

        # Spawn new fires based on state.
        rate = 0.0
        if status == "thinking":
            rate = 6.0
        elif status == "listening":
            rate = 1.0
        elif status == "speaking":
            rate = 2.0
        elif status == "idle":
            rate = 0.25
        if random.random() < rate * dt:
            i, j = random.choice(self.edges)
            self.fires.append({"i": i, "j": j, "t": 0.0})
            self.node_flash[i] = 0.8

        if status == "idle" and random.random() < 1.5 * dt:
            self.node_flash[random.randrange(NODE_COUNT)] = 0.6

        # Audio feedback keeps the waveform + sphere in sync.
        _ = audio_level

    def _neighbours_of(self, node: int) -> list[int]:
        nbrs = []
        for i, j in self.edges:
            if i == node:
                nbrs.append(j)
            elif j == node:
                nbrs.append(i)
        return nbrs or [random.randrange(NODE_COUNT)]

    @staticmethod
    def _spin_speed(status: str) -> float:
        if status == "thinking":
            return 1.4
        if status == "speaking":
            return 1.0
        if status == "listening":
            return 0.7
        return 0.35

    # ── Queryable per-frame state (used by both renderers) ─────────────────

    def sphere_transform(self, status: str, audio_level: float) -> tuple[np.ndarray, float, float]:
        """Return ``(rotated_nodes, scale, y_offset)`` for this frame."""
        scale = 1.0
        if status == "listening":
            scale = 1.0 + 0.055 * math.sin(self.t * 3.0)
        elif status == "thinking":
            scale = 1.03
        elif status == "speaking":
            scale = 1.0 + 0.025 * audio_level
        y_offset = 0.0
        if status == "speaking":
            y_offset = 0.22 * math.sin(self.t * 5.5) * max(0.15, audio_level)
        pts = _rot_y(self.nodes, self.yaw)
        return pts * scale + np.array([0.0, y_offset, 0.0]), scale, y_offset

    def ring_geometry(self) -> list[dict]:
        """Return per-ring rotated point sets + label anchor for this frame."""
        out = []
        for (radius, tilt, speed, phase, colour), label in zip(RING_PARAMS, RING_LABELS):
            angle = phase + self.t * speed
            pts = _rot_y(_ring_points(radius, tilt, angle), self.t * speed * 0.35 + phase)
            # Label anchor rides on the ring edge (angle offset 45 deg).
            anchor_angle = angle + math.radians(45)
            anchor = _rot_y(_ring_points(radius, tilt, anchor_angle), self.t * speed * 0.35 + phase)[0]
            out.append({"pts": pts, "colour": colour, "label": label, "anchor": anchor})
        return out

    def node_brightness(self) -> np.ndarray:
        """Per-node brightness in [0, 1] (base pulse + flash)."""
        base = 0.45 + 0.25 * np.sin(self.t * 0.6 + self.phase)
        return np.clip(base + self.node_flash, 0.0, 1.0)

    def fire_positions(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Return ``(start, end, t)`` for active sparks (world space)."""
        pts = _rot_y(self.nodes, self.yaw)
        out = []
        for fire in self.fires:
            out.append((pts[fire["i"]], pts[fire["j"]], fire["t"]))
        return out


# ── OpenGL availability probe ─────────────────────────────────────────────────

def _gl_available() -> bool:
    try:
        from OpenGL.GL import glBegin  # noqa: F401

        return True
    except Exception:
        return False


_GL_OK = _gl_available()

if _GL_OK:
    from OpenGL.GL import *  # noqa: F401,F403
    from OpenGL.GLU import gluPerspective, gluProject
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    class SphereGLWidget(QOpenGLWidget):  # type: ignore[misc]
        """OpenGL renderer for the neural sphere, rings and grid floor."""

        def __init__(self, sim: Sphere3DWidget, state=None) -> None:
            super().__init__()
            self.sim = sim
            self.state = state
            self.setMinimumSize(420, 360)
            self._status = "idle"
            self._audio = 0.0
            self._dt = 0.0

        # ── Qt OpenGL hooks ────────────────────────────────────────────────

        def initializeGL(self) -> None:
            glClearColor(*BACKGROUND)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # additive glow
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        def resizeGL(self, w: int, h: int) -> None:
            if h <= 0:
                h = 1
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(50.0, w / h, 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)

        def update_frame(self, dt: float, status: str, audio: float) -> None:
            self._dt = dt
            self._status = status
            self._audio = audio
            self.sim.update(dt, status, audio)
            self.update()

        # ── Rendering ─────────────────────────────────────────────────────

        def paintGL(self) -> None:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            eye = CAMERA_EYE.copy()
            if self.state is not None and self.state.glitch_active:
                eye[0] += random.uniform(-0.06, 0.06)
            from OpenGL.GLU import gluLookAt

            gluLookAt(
                eye[0], eye[1], eye[2],
                CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2],
                CAMERA_UP[0], CAMERA_UP[1], CAMERA_UP[2],
            )

            self._draw_grid()
            self._draw_sphere()
            self._draw_rings()
            self._draw_glow_overlay()
            self.paint_text_overlay()

        def _draw_grid(self) -> None:
            scroll = (self.sim.t * 0.9) % GRID_SPACING
            y = GRID_Y
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for i in range(-int(GRID_EXTENT), int(GRID_EXTENT) + 1):
                z = i + scroll
                fade = max(0.0, 1.0 - abs(z) / GRID_EXTENT)
                r = GRID_COLOR[0] + (GRID_FAR_COLOR[0] - GRID_COLOR[0]) * (1 - fade)
                g = GRID_COLOR[1] + (GRID_FAR_COLOR[1] - GRID_COLOR[1]) * (1 - fade)
                b = GRID_COLOR[2] + (GRID_FAR_COLOR[2] - GRID_COLOR[2]) * (1 - fade)
                glColor4f(r, g, b, 0.18 * fade + 0.02)
                glVertex3f(-GRID_EXTENT, y, z)
                glVertex3f(GRID_EXTENT, y, z)
            for i in range(-int(GRID_EXTENT), int(GRID_EXTENT) + 1):
                x = i
                fade = max(0.0, 1.0 - abs(x) / GRID_EXTENT)
                glColor4f(GRID_COLOR[0], GRID_COLOR[1], GRID_COLOR[2], 0.10 * fade)
                glVertex3f(x, y, -GRID_EXTENT)
                glVertex3f(x, y, GRID_EXTENT)
            glEnd()

        def _draw_sphere(self) -> None:
            pts, scale, y_offset = self.sim.sphere_transform(self._status, self._audio)
            glLineWidth(1.0)

            # Hologram wireframe body (great circles).
            glColor4f(*SPHERE_GRID)
            glBegin(GL_LINE_LOOP)
            for ring in self.sim.lat_rings:
                for v in _rot_y(ring, self.sim.yaw) * scale + np.array([0.0, y_offset, 0.0]):
                    glVertex3f(*v)
            glEnd()
            glBegin(GL_LINE_LOOP)
            for ring in self.sim.lon_rings:
                for v in _rot_y(ring, self.sim.yaw) * scale + np.array([0.0, y_offset, 0.0]):
                    glVertex3f(*v)
            glEnd()

            # Edges (semi-transparent cyan).
            glBegin(GL_LINES)
            glColor4f(*EDGE_COLOR)
            for i, j in self.sim.edges:
                glVertex3f(*pts[i])
                glVertex3f(*pts[j])
            glEnd()

            # Firing sparks along edges.
            glPointSize(3.0)
            glBegin(GL_POINTS)
            for start, end, t in self.sim.fire_positions():
                mid = start + (end - start) * t
                glColor4f(FIRE_COLOR[0], FIRE_COLOR[1], FIRE_COLOR[2], 0.9)
                glVertex3f(*mid)
            glEnd()

            # Node cores.
            brightness = self.sim.node_brightness()
            glPointSize(3.5)
            glBegin(GL_POINTS)
            for v, b in zip(pts, brightness):
                glColor4f(NODE_CORE[0], NODE_CORE[1], NODE_CORE[2], min(1.0, b))
                glVertex3f(*v)
            glEnd()

        def _draw_rings(self) -> None:
            for ring in self.sim.ring_geometry():
                pts, colour = ring["pts"], ring["colour"]
                glLineWidth(1.4)
                glBegin(GL_LINE_LOOP)
                glColor4f(colour[0], colour[1], colour[2], colour[3] * 0.55)
                for v in pts:
                    glVertex3f(*v)
                glEnd()
                # Bright comet head (light trail) at the leading edge.
                head = 20
                glBegin(GL_LINE_STRIP)
                for v in pts[:head]:
                    glColor4f(colour[0], colour[1], colour[2], colour[3])
                    glVertex3f(*v)
                glEnd()
                # Icon nodes at the four cardinal points.
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for v in pts[:: RING_SEGMENTS // 4]:
                    glColor4f(1.0, 1.0, 1.0, 0.9)
                    glVertex3f(*v)
                glEnd()

        def _draw_glow_overlay(self) -> None:
            """Billboarded additive glow quads around each node."""
            pts, scale, y_offset = self.sim.sphere_transform(self._status, self._audio)
            brightness = self.sim.node_brightness()

            # Camera basis for screen-aligned billboards.
            fwd = CAMERA_TARGET - CAMERA_EYE
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, CAMERA_UP)
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)

            glDepthMask(GL_FALSE)  # additive glows never occlude
            glBegin(GL_QUADS)
            for v, b in zip(pts, brightness):
                if b <= 0.02:
                    continue
                size = 0.045 + 0.085 * b
                alpha = min(0.9, 0.25 + b * 0.65)
                c = NODE_COLOR
                for u, vv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    corner = v + right * size * u + up * size * vv
                    glColor4f(c[0], c[1], c[2], alpha)
                    glVertex3f(*corner)
            glEnd()
            glDepthMask(GL_TRUE)

        # ── Text overlay (ring labels) via QPainter ───────────────────────

        def _project(self, x: float, y: float, z: float) -> Optional[tuple[float, float]]:
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            wx, wy, wz = gluProject(x, y, z, modelview, projection, viewport)
            if wz > 1.0:
                return None
            h = self.height()
            return float(wx), h - float(wy)

        def paint_text_overlay(self) -> None:
            """Draw ring labels as a 2D overlay (called from paintGL)."""
            from PySide6.QtGui import QPainter, QColor, QFont
            from PySide6.QtCore import Qt

            painter = QPainter(self)
            try:
                painter.setRenderHint(QPainter.Antialiasing)
                font = QFont("Courier New", 8, QFont.Bold)
                painter.setFont(font)
                for ring in self.sim.ring_geometry():
                    anchor = ring["anchor"]
                    pos = self._project(anchor[0], anchor[1], anchor[2])
                    if pos is None:
                        continue
                    x, y = pos
                    label = ring["label"]
                    fm = painter.fontMetrics()
                    tw = fm.horizontalAdvance(label)
                    th = fm.height()
                    # Dark backing for legibility.
                    painter.fillRect(x - tw / 2 - 3, y - th - 4, tw + 6, th + 3, QColor(0, 0, 0, 170))
                    painter.setPen(QColor(120, 220, 255, 240))
                    painter.drawText(int(x - tw / 2), int(y - 3), label)
            finally:
                painter.end()

    def _make_sphere_widget(sim: Sphere3DWidget, state) -> SphereGLWidget:
        return SphereGLWidget(sim, state)


else:
    # Software fallback — QPainter 2D projection of the same simulation.
    from PySide6.QtWidgets import QWidget
    from PySide6.QtGui import QPainter, QColor, QPen
    from PySide6.QtCore import Qt, QPointF

    class SphereGLWidget(QWidget):  # type: ignore[misc,no-redef]
        def __init__(self, sim: Sphere3DWidget, state=None) -> None:
            super().__init__()
            self.sim = sim
            self.state = state
            self.setMinimumSize(420, 360)
            self._status = "idle"
            self._audio = 0.0
            self._dt = 0.0

        def update_frame(self, dt: float, status: str, audio: float) -> None:
            self._dt = dt
            self._status = status
            self._audio = audio
            self.sim.update(dt, status, audio)
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            w, h = self.width(), self.height()
            cx, cy = w / 2, h / 2

            def proj(p: np.ndarray) -> QPointF:
                z = p[2]
                f = 1.0 / (1.0 + max(0.0, z * 0.16))
                return QPointF(cx + p[0] * 90 * f, cy - p[1] * 90 * f)

            pts, scale, y_offset = self.sim.sphere_transform(self._status, self._audio)
            pen = QPen(QColor(5, 60, 130, 40))
            pen.setWidth(1)
            painter.setPen(pen)
            for i, j in self.sim.edges:
                painter.drawLine(proj(pts[i]), proj(pts[j]))

            brightness = self.sim.node_brightness()
            for v, b in zip(pts, brightness):
                p = proj(v)
                r = 1.5 + 3.0 * b
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, int(220 * b) + 30, 255, int(255 * b)))
                painter.drawEllipse(p, r, r)

            painter.end()

    def _make_sphere_widget(sim: Sphere3DWidget, state) -> SphereGLWidget:  # type: ignore[misc]
        return SphereGLWidget(sim, state)


def make_sphere_widget(sim: Sphere3DWidget, state=None) -> SphereGLWidget:
    """Factory used by ``ui_main`` so the GL/software decision stays local."""
    return _make_sphere_widget(sim, state)