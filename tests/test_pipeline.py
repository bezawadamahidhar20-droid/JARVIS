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

    def ask(self, user_input, memory=None, context=None):
        self.asked.append(user_input)
        return self.reply

    def ask_stream(self, user_input, memory=None, on_sentence=None, context=None):
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
    assert jarvis.process_input("what time is it")
    assert any("It's" in s for s in jarvis.tts.spoken)


def test_normal_command_open_app_full_pipeline(monkeypatch):
    launched = []
    monkeypatch.setattr(sc.subprocess, "Popen", lambda cmd: launched.append(cmd))
    jarvis = make_jarvis()
    assert jarvis.process_input("open notepad")
    assert launched == ["notepad.exe"]
    assert any("Opening notepad" in s for s in jarvis.tts.spoken)


def test_deterministic_commands_bypass_llm(monkeypatch):
    """Issue: simple deterministic commands MUST NOT use the LLM.

    Pipeline is Whisper → Router → Tool (no Qwen3 round-trip), so
    commands like "open Notepad" or "what time is it" stay near
    instant even when the AI model is slow.
    """
    launched = []
    monkeypatch.setattr(sc.subprocess, "Popen", lambda cmd: launched.append(cmd))
    provider = FakeProvider(reply="should never be used")
    jarvis = make_jarvis(provider=provider)

    for cmd in (
        "open notepad",
        "open calculator",
        "what time is it",
        "what is today's date",
    ):
        assert jarvis.process_input(cmd)

    # The AI provider was never consulted for any deterministic command.
    assert provider.asked == []
    # The tool layer actually ran (apps launched via the registry).
    assert launched == ["notepad.exe", "calc.exe"]
    assert any("It's" in s for s in jarvis.tts.spoken)  # clock answer
    assert any("Today is" in s for s in jarvis.tts.spoken)  # calendar


# ── Runtime model-mode switch ────────────────────────────────

class SwitchableProvider(FakeProvider):
    """FakeProvider that supports runtime model switches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "qwen3:8b"
        self.switched = []

    def switch_model(self, model):
        self.switched.append(model)
        self.model = model
        return model


def _switch_jarvis(monkeypatch, provider=None):
    from config import jarvis_config

    # No background warm-up threads in tests.
    monkeypatch.setattr(jarvis_config, "ENABLE_WARMUP", False)
    return make_jarvis(
        provider=provider if provider is not None else SwitchableProvider()
    )


def test_runtime_switch_to_fast_mode(monkeypatch):
    """"Switch to fast mode" must change the provider's model without
    an LLM round-trip and without a restart."""
    provider = SwitchableProvider()
    jarvis = _switch_jarvis(monkeypatch, provider)
    assert jarvis.process_input("switch to fast mode")
    assert provider.model == "qwen3:1.7b"
    assert provider.asked == []  # deterministic — no LLM call
    assert any("fast" in s.lower() and "qwen3:1.7b" in s
               for s in jarvis.tts.spoken)


def test_runtime_switch_to_quality_mode(monkeypatch):
    provider = SwitchableProvider()
    jarvis = _switch_jarvis(monkeypatch, provider)
    # Start in fast mode, then switch to quality.
    assert jarvis.process_input("switch to fast mode")
    assert provider.model == "qwen3:1.7b"
    assert jarvis.process_input("switch to quality mode")
    assert provider.model == "qwen3:8b"
    assert provider.switched == ["qwen3:1.7b", "qwen3:8b"]


def test_runtime_switch_idempotent(monkeypatch):
    """Switching to the mode already active is a friendly no-op."""
    provider = SwitchableProvider()
    jarvis = _switch_jarvis(monkeypatch, provider)
    assert jarvis.process_input("switch to quality mode")
    assert provider.model == "qwen3:8b"
    assert provider.switched == []  # same model — nothing to switch
    assert any("already" in s.lower() for s in jarvis.tts.spoken)


def test_runtime_model_status_query(monkeypatch):
    """"Which model are you using" reports the active model — no LLM."""
    provider = SwitchableProvider()
    jarvis = _switch_jarvis(monkeypatch, provider)
    assert jarvis.process_input("which model are you using")
    assert provider.asked == []
    assert any("qwen3:8b" in s and "quality" in s for s in jarvis.tts.spoken)


def test_runtime_switch_unknown_mode_falls_back_to_llm(monkeypatch):
    """An unknown mode is a normal question, not a crash."""
    provider = SwitchableProvider()
    jarvis = _switch_jarvis(monkeypatch, provider)
    assert jarvis.process_input("switch to turbo mode")
    assert provider.asked  # routed to the AI brain
    assert provider.model == "qwen3:8b"  # unchanged


# ── Invalid input ─────────────────────────────────────────────

def test_invalid_input_does_not_crash():
    jarvis = make_jarvis()
    for bad in ("", "   ", "\t\n", "x" * (MAX_INPUT_CHARS + 1)):
        result = jarvis.process_input(bad)
        assert result is None or isinstance(result, str)


# ── Confirmation-required command ─────────────────────────────

def test_confirmation_required_command_flow(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()

    # Ask: nothing executes yet, a confirmation prompt is spoken.
    assert jarvis.process_input("shut down my computer")
    assert runs == []
    assert any("continue" in s.lower() for s in jarvis.tts.spoken)

    # Confirm: executes exactly once.
    jarvis.tts.spoken.clear()
    assert jarvis.process_input("yes")
    assert len(runs) == 1
    assert runs[0][0] == "shutdown"


def test_confirmation_rejected_never_executes(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    assert jarvis.process_input("restart my computer")
    assert jarvis.process_input("no")
    assert runs == []


# ── LLM fallback (provider offline) ───────────────────────────

def test_llm_fallback_when_provider_offline():
    jarvis = make_jarvis(provider=FakeProvider(available=False))
    assert jarvis.ollama_ok is False
    # AI question -> spoken offline message, no crash.
    assert jarvis.process_input("what is python")
    assert any("offline" in s.lower() for s in jarvis.tts.spoken)


# ── TTS cancellation via router ───────────────────────────────

def test_stop_speaking_cancels_tts():
    jarvis = make_jarvis()
    jarvis.process_input("stop speaking")
    assert jarvis.tts.stops == 1


# ── Memory flows through the pipeline ─────────────────────────

def test_conversation_memory_flows():
    provider = FakeProvider(reply="It is a language.")
    jarvis = make_jarvis(provider=provider)
    jarvis.process_input("what is python")
    assert len(jarvis.memory) == 2  # user + assistant stored
    jarvis.process_input("clear memory")
    assert len(jarvis.memory) == 0
