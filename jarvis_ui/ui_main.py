"""JARVIS UI — main window and 60fps animation loop.

Run with::

    python jarvis_ui/ui_main.py

The window is organised as:

    +--------------------------------------------------+
    |  TOP BAR: JARVIS logo + status + time + date      |
    +--------------------------------------------------+
    | LEFT PANEL |   CENTER 3D SPHERE + WAVEFORM   | RIGHT |
    | (AI NEURAL |   (OpenGL hologram)            | (CORE)|
    +--------------------------------------------------+
    |  BOTTOM BAR: mic + text input + quick commands   |
    +--------------------------------------------------+

Keyboard shortcuts:
    SPACE   toggle microphone
    ESC     exit JARVIS
    C       clear memory
    T       focus the text input
"""

from __future__ import annotations

import math
import queue
import sys
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QFont, QPainter, QColor, QShortcut, QKeySequence, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
)

from jarvis_ui.assets.fonts import pick_font_family
from jarvis_ui.ui_state import (
    JARVISState,
    JARVISController,
    SystemStatsThread,
    AudioMonitorThread,
    IDLE,
    LISTENING,
    THINKING,
    SPEAKING,
    SHUTDOWN,
)
from jarvis_ui.widgets.sphere_3d import Sphere3DWidget, make_sphere_widget
from jarvis_ui.widgets.waveform import WaveformWidget
from jarvis_ui.widgets.particles import ParticleLayer
from jarvis_ui.widgets.hud_panels import LeftPanel, RightPanel
from jarvis_ui.widgets.status_bar import TopStatusBar, BottomBar
from jarvis_ui.animations.boot_sequence import BootSequence
from jarvis_ui.animations.scan_line import ScanLineOverlay, GlitchOverlay
from jarvis_ui.animations.transitions import FadeLayer

FPS = 60
FRAME_MS = int(1000 / FPS)
FONT_FAMILY = pick_font_family()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S — AI Assistant")
        self.setMinimumSize(1100, 720)

        # ── State + backend ────────────────────────────────────────────────
        self.state = JARVISState()
        self.ui_queue: queue.Queue = queue.Queue()
        self.input_queue: queue.Queue = queue.Queue()
        self.controller = JARVISController(self.state, self.ui_queue, self.input_queue)
        self.stats_thread = SystemStatsThread(self.state, self.ui_queue)
        self.audio_thread = AudioMonitorThread(self.state)

        # ── Central background (particles) ─────────────────────────────────
        self.central = ParticleLayer(self.state)
        self.setCentralWidget(self.central)

        root = QVBoxLayout(self.central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top bar ────────────────────────────────────────────────────────
        self.top_bar = TopStatusBar(self.state)
        root.addWidget(self.top_bar)

        # ── Middle ─────────────────────────────────────────────────────────
        middle = QHBoxLayout()
        middle.setSpacing(8)

        self.left_panel = LeftPanel(self.state)
        middle.addWidget(self.left_panel, 0)

        center_col = QVBoxLayout()
        center_col.setSpacing(8)
        self.sim = Sphere3DWidget()
        self.sphere = make_sphere_widget(self.sim, self.state)
        self.waveform = WaveformWidget(self.state)
        center_col.addWidget(self.sphere, 3)
        center_col.addWidget(self.waveform, 1)
        middle.addLayout(center_col, 3)

        self.right_panel = RightPanel(self.state)
        middle.addWidget(self.right_panel, 0)

        root.addLayout(middle, 1)

        # ── Bottom bar ─────────────────────────────────────────────────────
        self.bottom_bar = BottomBar(
            self.state,
            on_send=self._on_send,
            on_mic_toggle=self._on_mic_toggle,
            on_quick=self._on_quick,
        )
        root.addWidget(self.bottom_bar)

        # ── Overlays ───────────────────────────────────────────────────────
        self.scan_line = ScanLineOverlay(self.central, self.state)
        self.glitch = GlitchOverlay(self.central, self.state)
        self.fade = FadeLayer(self.central)
        self.boot = BootSequence(self.central, on_finished=self._on_boot_done)

        self._resize_overlays()

        # ── Keyboard shortcuts ─────────────────────────────────────────────
        self._bind_shortcuts()

        # ── Animation loop ─────────────────────────────────────────────────
        self._last_tick = time.perf_counter()
        self._speak_start = 0.0
        self._prev_status = IDLE

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)

        self.boot.show()
        self.boot.raise_()

    # ── Layout / resize ────────────────────────────────────────────────────

    def _resize_overlays(self) -> None:
        rect = self.central.rect()
        for overlay in (self.scan_line, self.glitch, self.fade, self.boot):
            overlay.setGeometry(rect)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._resize_overlays()

    # ── Shortcuts ──────────────────────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        space = QShortcut(QKeySequence(Qt.Key_Space), self)
        space.activated.connect(self._toggle_mic)
        esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc.activated.connect(self._shutdown)
        c = QShortcut(QKeySequence("C"), self)
        c.activated.connect(lambda: self._on_quick("CLEAR MEMORY"))
        t = QShortcut(QKeySequence("T"), self)
        t.activated.connect(lambda: self.bottom_bar.input.setFocus())

    # ── Backend events ─────────────────────────────────────────────────────

    def _on_boot_done(self) -> None:
        self.fade.flash(duration=0.45)
        self.controller.start()
        self.stats_thread.start()
        self.audio_thread.start()

    def _on_send(self, text: str) -> None:
        self.input_queue.put(text)
        self.state.set_audio_level(0.0)

    def _on_quick(self, label: str) -> None:
        commands = {
            "TIME": "what time is it",
            "DATE": "what's the date",
            "YOUTUBE": "open youtube",
            "CLEAR MEMORY": "clear memory",
        }
        self.input_queue.put(commands[label])

    def _on_mic_toggle(self, active: bool) -> None:
        self.state.mic_enabled = active
        self.state.set_audio_level(0.0)

    def _toggle_mic(self) -> None:
        if self.bottom_bar.input.hasFocus():
            return
        active = not self.state.mic_enabled
        self.state.mic_enabled = active
        self.bottom_bar.mic.set_active(active)

    # ── Main loop ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now

        self._drain_queue()

        status = self.state.get_status()
        self.state.thinking_progress = 1.0 if status == THINKING else 0.0

        if status != self._prev_status:
            if status == SPEAKING:
                self._speak_start = now
            self._prev_status = status

        # Audio level: mic level while listening, synthetic voice while speaking.
        if status == SPEAKING:
            level = self._synthetic_speech_level(now)
        else:
            level = self.state.get_audio_level()
            if status == LISTENING:
                level = max(level, self.state.get_audio_level())

        # Update visuals.
        self.sphere.update_frame(dt, status, level)
        self.waveform.update_frame(dt, status, level)
        self.left_panel.tick(dt)
        self.right_panel.tick(dt)
        self.top_bar.tick(dt)
        self.bottom_bar.tick(dt)
        self.scan_line.tick(dt)
        self.glitch.tick(dt)
        self.central.update_frame(dt, status, self.central.width(), self.central.height())

        if status == SHUTDOWN:
            self._shutdown()

    def _synthetic_speech_level(self, now: float) -> float:
        """Voice-like envelope so the waveform reacts while JARVIS speaks."""
        elapsed = now - self._speak_start
        response = self.state.current_response or ""
        duration = max(0.8, len(response) * 0.045)
        if elapsed > duration:
            return 0.0
        envelope = math.sin(math.pi * min(1.0, elapsed / duration)) ** 2
        voice = 0.45 + 0.55 * math.sin(2 * math.pi * 3.0 * elapsed)
        return max(0.0, min(1.0, envelope * voice))

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self.ui_queue.get_nowait()
            except queue.Empty:
                return
            kind = event.get("event")
            if kind == "status_change":
                self.state.set_status(event.get("value", IDLE))
            elif kind == "input":
                self.state.current_input = event.get("text", "")
            elif kind == "response":
                self.state.current_response = event.get("text", "")
            elif kind == "module":
                self.state.set_module(event.get("name", ""), event.get("active", False))
            elif kind == "stats":
                self.state.cpu_usage = event.get("cpu", 0.0)
                self.state.ram_usage = event.get("ram", 0.0)
            elif kind == "memory_cleared":
                self.state.clear_conversation()
            elif kind == "log":
                print(f"[UI] {event.get('text', '')}")

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        self.controller.stop()
        self.stats_thread.stop()
        self.audio_thread.stop()
        self.timer.stop()
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._shutdown()
        super().closeEvent(event)


def main() -> int:
    # Request an OpenGL compatibility profile so fixed-function immediate-mode
    # rendering (glBegin/glEnd) works everywhere.
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("J.A.R.V.I.S")
    app.setStyle("Fusion")

    font = QFont(FONT_FAMILY)
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())