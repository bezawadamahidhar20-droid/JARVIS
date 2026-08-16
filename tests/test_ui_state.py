"""GUI backend controller tests (no PySide6, no hardware).

jarvis_ui/ui_state.py's JARVISController runs the real pipeline in a
background thread. These tests verify the wiring (imports, intent
constants, config attribute names) without opening a window — they
guard against the pre-existing regressions that silently forced the
GUI into demo mode.
"""

import queue

from jarvis_ui.ui_state import JARVISController, JARVISState


def _make_controller():
    return JARVISController(
        JARVISState(), queue.Queue(), queue.Queue()
    )


def test_import_backend_succeeds():
    """The real backend modules must import cleanly — the previous
    broken imports (e.g. `from brain.router import AI_QUESTION`) made
    the GUI fall back to demo mode unconditionally."""
    ctrl = _make_controller()
    assert ctrl._import_backend() is True


class FakeTTS:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


class FakeOllama:
    def __init__(self, reply="A concise answer."):
        self.reply = reply

    def ask(self, user_input, memory=None):
        return self.reply


class FakeCommands:
    def execute(self, text):
        return "command response"


class FakeRouter:
    def route(self, text):
        from brain.router import Intent

        if text == "what time is it":
            return Intent.COMMAND, text
        return Intent.AI_QUESTION, text


class FakeMemory:
    def add_user_message(self, text):
        pass

    def add_assistant_message(self, text):
        pass

    def clear(self):
        pass


def test_process_text_command_path():
    ctrl = _make_controller()
    ctrl.router = FakeRouter()
    ctrl.commands = FakeCommands()
    ctrl.tts = FakeTTS()
    ctrl.memory = FakeMemory()
    ctrl.ollama = FakeOllama()

    ctrl._process_text("what time is it")
    assert ctrl.state.add_command_history  # command recorded
    assert "command response" in ctrl.state.current_response


def test_process_text_ai_path():
    ctrl = _make_controller()
    ctrl.router = FakeRouter()
    ctrl.commands = FakeCommands()
    ctrl.tts = FakeTTS()
    ctrl.memory = FakeMemory()
    ctrl.ollama = FakeOllama(reply="Recursion is a function that calls itself.")

    ctrl._process_text("explain recursion")
    assert "Recursion" in ctrl.state.current_response


def test_process_text_ai_failure_shows_error_state():
    class BoomOllama:
        def ask(self, user_input, memory=None):
            raise RuntimeError("Ollama down")

    ui_queue = queue.Queue()
    ctrl = JARVISController(JARVISState(), ui_queue, queue.Queue())
    ctrl.router = FakeRouter()
    ctrl.commands = FakeCommands()
    ctrl.tts = FakeTTS()
    ctrl.memory = FakeMemory()
    ctrl.ollama = BoomOllama()

    ctrl._process_text("explain recursion")
    assert ctrl.state.ollama_connected is False
    assert "offline" in ctrl.state.current_response.lower()

    # The ERROR status event reached the UI state machine (issue 14).
    from jarvis_ui.ui_state import ERROR

    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())
    assert {"event": "status_change", "value": ERROR} in events
