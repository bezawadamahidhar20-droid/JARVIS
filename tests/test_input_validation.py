"""Issue 1 — input sanitisation and length validation.

Normal commands must pass through unchanged; empty and abnormally long
input must be rejected safely without crashing JARVIS.
"""

from brain.router import (
    Intent,
    IntentRouter,
    MAX_INPUT_CHARS,
    sanitize_input,
    validate_input,
)
from brain.memory import ConversationMemory
from commands.registry import CommandRegistry
from main import JARVIS


# ── Unit: sanitize / validate ─────────────────────────────────

def test_sanitize_collapses_whitespace():
    assert sanitize_input("  open   chrome  ") == "open chrome"
    assert sanitize_input("hello\nworld") == "hello world"
    assert sanitize_input("") == ""
    assert sanitize_input(None) == ""
    assert sanitize_input("   ") == ""


def test_validate_accepts_normal_command():
    assert validate_input("open chrome") == "open chrome"


def test_validate_rejects_empty():
    assert validate_input("") == ""
    assert validate_input("   \n\t ") == ""
    assert validate_input(None) == ""


def test_validate_rejects_overlong_input():
    long_input = "a" * (MAX_INPUT_CHARS + 1)
    assert validate_input(long_input) == ""
    # Exactly at the limit is accepted.
    assert validate_input("b" * MAX_INPUT_CHARS) == "b" * MAX_INPUT_CHARS


def test_validate_accepts_unicode():
    assert validate_input("qué hora es") == "qué hora es"
    assert validate_input("מה השעה") == "מה השעה"


# ── Router level ──────────────────────────────────────────────

def test_router_accepts_normal_command():
    router = IntentRouter()
    intent, _ = router.route("open chrome")
    assert intent == Intent.COMMAND


def test_router_rejects_overlong_input():
    router = IntentRouter()
    intent, cleaned = router.route("open chrome " + "x" * MAX_INPUT_CHARS)
    assert intent == Intent.UNKNOWN
    assert cleaned == ""


def test_router_rejects_whitespace_only():
    router = IntentRouter()
    intent, _ = router.route("   \n  ")
    assert intent == Intent.UNKNOWN


def test_router_rejects_empty():
    router = IntentRouter()
    intent, _ = router.route("")
    assert intent == Intent.UNKNOWN


def test_router_never_crashes_on_control_chars():
    router = IntentRouter()
    intent, _ = router.route("\x00\x01\x02 open chrome \x7f")
    assert intent in (Intent.COMMAND, Intent.UNKNOWN)


# ── Full pipeline (process_input) ─────────────────────────────

class FakeTTS:
    def __init__(self):
        self.spoken = []

    def load(self):
        return True

    def speak(self, text):
        if text:
            self.spoken.append(text)

    def speak_blocking(self, text):
        self.speak(text)

    def wait(self):
        pass

    def stop(self):
        pass


class FakeSTT:
    vad = None

    def load(self):
        return True

    def listen(self):
        return None

    def unload(self):
        pass


class FakeMic:
    def is_available(self):
        return True

    def describe(self):
        return "fake mic"


class FakeProvider:
    name = "fake"

    def __init__(self, reply="A concise answer."):
        self.reply = reply
        self.asked = []

    def is_available(self):
        return True

    def ask(self, user_input, memory=None):
        self.asked.append(user_input)
        return self.reply

    def ask_stream(self, user_input, memory=None, on_sentence=None):
        self.asked.append(user_input)
        if on_sentence:
            on_sentence(self.reply)
        return self.reply

    def describe(self):
        return "fake provider"


def make_jarvis():
    return JARVIS(
        text_mode=True,
        components={
            "mic": FakeMic(),
            "stt": FakeSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000, persist_path=""),
            "provider": FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
        },
    )


def test_pipeline_normal_command_works():
    jarvis = make_jarvis()
    assert jarvis.process_input("open chrome") is True
    assert jarvis.tts.spoken  # a spoken response, not a crash


def test_pipeline_overlong_input_rejected_safely():
    jarvis = make_jarvis()
    assert jarvis.process_input("hello " + "x" * MAX_INPUT_CHARS) is True
    # The polite rejection was spoken; nothing was routed to the AI.
    assert any("too long" in s.lower() for s in jarvis.tts.spoken)
    assert jarvis.provider.asked == []


def test_pipeline_empty_and_whitespace_ignored():
    jarvis = make_jarvis()
    assert jarvis.process_input("") is True
    assert jarvis.process_input("   ") is True
    assert jarvis.process_input("\t\n") is True
    assert jarvis.provider.asked == []
