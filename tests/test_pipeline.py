"""Issue 11 — full-pipeline integration tests.

voice/input → router → command/LLM → response, with every hardware and
external service mocked. No microphone, speakers, live Ollama, or live
Groq are required.
"""

import commands.system_commands as sc
from brain.memory import ConversationMemory
from brain.router import IntentRouter, MAX_INPUT_CHARS
from commands.registry import CommandRegistry
from main import JARVIS


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.stops = 0

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
        self.stops += 1


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

    def __init__(self, available=True, reply="A concise answer."):
        self.available = available
        self.reply = reply
        self.asked = []

    def is_available(self):
        return self.available

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


def make_jarvis(provider=None, text_mode=True):
    return JARVIS(
        text_mode=text_mode,
        components={
            "mic": FakeMic(),
            "stt": FakeSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000, persist_path=""),
            "provider": provider if provider is not None else FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
        },
    )


# ── Normal command ────────────────────────────────────────────

def test_normal_command_full_pipeline():
    jarvis = make_jarvis()
    assert jarvis.process_input("what time is it") is True
    assert any("It's" in s for s in jarvis.tts.spoken)


def test_normal_command_open_app_full_pipeline(monkeypatch):
    launched = []
    monkeypatch.setattr(sc.subprocess, "Popen", lambda cmd: launched.append(cmd))
    jarvis = make_jarvis()
    assert jarvis.process_input("open notepad") is True
    assert launched == ["notepad.exe"]
    assert any("Opening notepad" in s for s in jarvis.tts.spoken)


# ── Invalid input ─────────────────────────────────────────────

def test_invalid_input_does_not_crash():
    jarvis = make_jarvis()
    for bad in ("", "   ", "\t\n", "x" * (MAX_INPUT_CHARS + 1)):
        assert jarvis.process_input(bad) is True


# ── Confirmation-required command ─────────────────────────────

def test_confirmation_required_command_flow(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()

    # Ask: nothing executes yet, a confirmation prompt is spoken.
    assert jarvis.process_input("shut down my computer") is True
    assert runs == []
    assert any("continue" in s.lower() for s in jarvis.tts.spoken)

    # Confirm: executes exactly once.
    jarvis.tts.spoken.clear()
    assert jarvis.process_input("yes") is True
    assert len(runs) == 1
    assert runs[0][0] == "shutdown"


def test_confirmation_rejected_never_executes(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    assert jarvis.process_input("restart my computer") is True
    assert jarvis.process_input("no") is True
    assert runs == []


# ── LLM fallback (provider offline) ───────────────────────────

def test_llm_fallback_when_provider_offline():
    jarvis = make_jarvis(provider=FakeProvider(available=False))
    assert jarvis.ollama_ok is False
    # AI question -> spoken offline message, no crash.
    assert jarvis.process_input("what is python") is True
    assert any("offline" in s.lower() for s in jarvis.tts.spoken)


# ── TTS cancellation via router ───────────────────────────────

def test_stop_speaking_cancels_tts():
    jarvis = make_jarvis()
    assert jarvis.process_input("stop speaking") is True
    assert jarvis.tts.stops == 1


# ── Memory flows through the pipeline ─────────────────────────

def test_conversation_memory_flows():
    provider = FakeProvider(reply="It is a language.")
    jarvis = make_jarvis(provider=provider)
    jarvis.process_input("what is python")
    assert len(jarvis.memory) == 2  # user + assistant stored
    jarvis.process_input("clear memory")
    assert len(jarvis.memory) == 0
