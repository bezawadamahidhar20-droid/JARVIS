"""GUI smoke tests — the PySide6 interface, verified headlessly.

PySide6 lives in the optional ``gui`` extra (jarvis_ui/requirements_ui.txt),
so these tests skip cleanly on machines without it. CI runs them in a
dedicated job with QT_QPA_PLATFORM=offscreen (no display needed).

What is covered (no real window, no OpenGL, no mic):
  * ``jarvis_ui.ui_main`` imports and exposes MainWindow + main().
  * ``JARVISState`` / ``JARVISController`` (the UI <-> backend bridge)
    construct and stop cleanly without a QApplication.
"""

import os
import queue

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed (gui extra)")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_module_imports_headless():
    """The GUI package must import without a display or audio device."""
    import jarvis_ui.ui_main as ui_main
    import jarvis_ui.ui_state as ui_state

    assert hasattr(ui_main, "MainWindow")
    assert callable(ui_main.main)
    assert hasattr(ui_state, "JARVISController")
    assert hasattr(ui_state, "IDLE")
    assert hasattr(ui_state, "THINKING")


def test_gui_state_and_controller_construct():
    """The UI <-> backend bridge must build and stop without Qt."""
    from jarvis_ui.ui_state import JARVISController, JARVISState

    state = JARVISState()
    controller = JARVISController(state, queue.Queue(), queue.Queue())

    state.set_status("thinking")
    assert state.get_status() == "thinking"

    controller.stop()  # must be idempotent and non-blocking
    controller.stop()


def test_gui_controller_runs_simulation_fallback(monkeypatch):
    """When the backend cannot be imported (no mic/Ollama install) the
    controller falls back to a demo loop that keeps the UI alive — it
    must start, push a greeting response, and stop cleanly."""
    from jarvis_ui.ui_state import JARVISController, JARVISState

    state = JARVISState()
    ui_queue = queue.Queue()
    controller = JARVISController(state, ui_queue, queue.Queue())
    monkeypatch.setattr(controller, "_import_backend", lambda: False)
    controller.start()
    # Give the sim loop a moment to push a greeting.
    import time

    deadline = time.time() + 5
    found = False
    while time.time() < deadline:
        try:
            event = ui_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.05)
            continue
        if event.get("event") == "response":
            found = True
            break
    controller.stop()
    assert found  # the greeting response arrived
