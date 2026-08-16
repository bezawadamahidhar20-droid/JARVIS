"""Streaming STT, wake word, and barge-in tests (all faked, no audio)."""

import asyncio
import importlib.util

import numpy as np
import pytest

import engine.stt as stt_mod
import sounddevice as sd
from brain.memory import ConversationMemory
from brain.router import Intent, IntentRouter
from commands.registry import CommandRegistry
from engine.barge_in import BargeInMonitor
from engine.wakeword import WakeWordDetector
from main import JARVIS


# ── Streaming STT: async plumbing ─────────────────────────────

def _make_stream_stt(monkeypatch):
    """STTEngine with streaming enabled and a scripted sync stream."""
    monkeypatch.setattr(stt_mod, "STT_STREAM", True)
    monkeypatch.setattr(stt_mod, "ENABLE_WAKE_WORD", False)
    stt = stt_mod.STTEngine()
    assert stt._streaming is not None

    def fake_stream():
        yield ("stop speaking", False)
        yield ("stop speaking", True)

    monkeypatch.setattr(stt, "stream_listen", fake_stream)
    return stt


async def _collect(agen):
    return [item async for item in agen]


def test_astream_listen_bridges_sync_stream(monkeypatch):
    stt = _make_stream_stt(monkeypatch)
    got = asyncio.run(_collect(stt.astream_listen()))
    assert got == [("stop speaking", False), ("stop speaking", True)]


def test_stream_listen_raises_when_disabled(monkeypatch):
    monkeypatch.setattr(stt_mod, "STT_STREAM", False)
    monkeypatch.setattr(stt_mod, "ENABLE_WAKE_WORD", False)
    stt = stt_mod.STTEngine()
    assert stt._streaming is None
    with pytest.raises(RuntimeError):
        list(stt.stream_listen())


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


class FakeStreamSTT:
    """STT fake exposing astream_listen + _streaming (streaming path)."""

    vad = None
    _streaming = object()

    def load(self):
        return True

    async def astream_listen(self):
        yield ("stop speaking", False)
        yield ("stop speaking", True)

    def unload(self):
        pass


class FakeMic:
    def is_available(self):
        return True

    def describe(self):
        return "fake mic"


class FakeProvider:
    name = "fake"

    def is_available(self):
        return True

    def ask(self, *_a, **_k):
        return None

    def ask_stream(self, *_a, **_k):
        return None

    def describe(self):
        return "fake"


def _jarvis_with_streaming_stt():
    return JARVIS(
        text_mode=False,
        components={
            "mic": FakeMic(),
            "stt": FakeStreamSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000, persist_path=""),
            "provider": FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
        },
    )


def test_listen_voice_acts_on_partial_stop_early():
    """A partial 'stop speaking' must interrupt TTS before the phrase
    even ends; the final text still drives normal routing."""
    jarvis = _jarvis_with_streaming_stt()
    result = jarvis._run_coro(jarvis._listen_voice())
    assert result == "stop speaking"  # final transcription returned
    assert jarvis.tts.stops == 1  # early action happened


def test_router_stop_intent_matches_partial():
    router = IntentRouter()
    intent, _ = router.route("stop speaking")
    assert intent == Intent.STOP_SPEECH


# ── Wake word: graceful fallback ──────────────────────────────

def test_wake_word_detector_falls_back_when_package_missing():
    if importlib.util.find_spec("openwakeword") is not None:
        pytest.skip("openwakeword installed — model download required")
    detector = WakeWordDetector(wake_word="hey jarvis")
    assert detector.load() is False
    assert detector.available is False
    assert detector.load_error is not None
    # Without the model, waiting must fail fast (never hang, never crash).
    assert detector.wait_for_wake_word(timeout=0.1) is False


def test_main_wake_word_helpers_noop_when_disabled():
    """With the default .env (wake word off) the loop helpers must be
    cheap no-ops."""
    from config import jarvis_config

    if jarvis_config.ENABLE_WAKE_WORD:
        pytest.skip("ENABLE_WAKE_WORD=true in this .env")

    jarvis = _jarvis_with_streaming_stt()
    assert jarvis._start_barge_in() is None  # barge-in off by default
    jarvis._run_coro(jarvis._await_wake_word())  # must not raise/hang


# ── Barge-in monitor ──────────────────────────────────────────

def test_barge_in_fires_on_speech(monkeypatch):
    fired = []
    # The monitor's callback receives float32 audio in [-1, 1], so the
    # threshold is set on that scale.
    monitor = BargeInMonitor(
        sample_rate=16000,
        threshold=0.5,
        chunks_to_confirm=2,
        chunk_duration=0.05,
    )

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import threading
    import time

    entered = threading.Event()

    def fake_input_stream(**kwargs):
        callback = kwargs["callback"]
        loud = np.full(kwargs["blocksize"], 0.9, dtype=np.float32)
        callback(loud, 0, None, None)
        callback(loud, 0, None, None)  # second loud chunk -> confirmed
        entered.set()
        return FakeStream()

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    monitor.start(on_speech=lambda: fired.append(True))
    # Wait until the monitor thread is actually inside the stream (the
    # callbacks fire synchronously on entry), then stop it.
    assert entered.wait(2.0)
    time.sleep(0.05)
    monitor.stop()
    assert fired, "on_speech must fire when loud audio is heard"
    assert monitor.active is False


def test_barge_in_ignores_silence(monkeypatch):
    fired = []
    monitor = BargeInMonitor(
        sample_rate=16000,
        threshold=0.5,
        chunks_to_confirm=3,
        chunk_duration=0.05,
    )

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_input_stream(**kwargs):
        callback = kwargs["callback"]
        quiet = np.zeros(kwargs["blocksize"], dtype=np.float32)
        for _ in range(10):
            callback(quiet, 0, None, None)
        return FakeStream()

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    monitor.start(on_speech=lambda: fired.append(True))
    monitor.stop()
    assert fired == []


def test_barge_in_requires_confirming_chunks(monkeypatch):
    """One loud chunk must not fire — the confirm count gates it."""
    fired = []
    monitor = BargeInMonitor(
        sample_rate=16000,
        threshold=0.5,
        chunks_to_confirm=3,
        chunk_duration=0.05,
    )

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_input_stream(**kwargs):
        callback = kwargs["callback"]
        loud = np.full(kwargs["blocksize"], 0.9, dtype=np.float32)
        callback(loud, 0, None, None)  # only one loud chunk
        callback(loud, 0, None, None)  # still below the confirm count
        return FakeStream()

    monkeypatch.setattr(sd, "InputStream", fake_input_stream)
    monitor.start(on_speech=lambda: fired.append(True))
    monitor.stop()
    assert fired == []
