"""TTS engine tests — fake backend injected, no real voice or audio."""

import threading

from engine.tts import TTSEngine, clean_for_speech, _resolve_voice_path


class FakeBackend:
    """Records spoken text; never touches real audio hardware."""

    def __init__(self, fail=False):
        self.spoken = []
        self.fail = fail

    def load(self) -> bool:
        return not self.fail

    def synthesize(self, text):
        if self.fail:
            raise RuntimeError("voice unavailable")
        return []

    def play(self, audio):
        if self.fail:
            raise RuntimeError("audio unavailable")
        self.spoken.append("played")


class RecordingBackend(FakeBackend):
    """Fake that exposes what the worker actually spoke."""

    def __init__(self):
        super().__init__()
        self.said = []

    def _speak(self, text):
        self.said.append(text)


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**bold** text") == "bold text"
    assert clean_for_speech("`code` here") == "code here"
    assert clean_for_speech("## Header") == "Header"
    assert clean_for_speech("[link](https://x.com)") == "link"
    assert clean_for_speech("a   b") == "a b"


def test_clean_for_speech_handles_empty():
    assert clean_for_speech("") == ""
    assert clean_for_speech("   ") == ""


def test_speak_queues_cleaned_text():
    backend = FakeBackend()
    tts = TTSEngine(engine="piper", backend=backend)
    tts.speak("**Hello** there")
    tts.wait()
    tts.stop()
    assert backend.spoken == ["played"]


def test_speak_with_failing_backend_never_raises():
    tts = TTSEngine(engine="piper", backend=FakeBackend(fail=True))
    tts.speak("This should still print")
    tts.wait()
    tts.stop()


def test_speak_blocking_returns_even_on_failure():
    tts = TTSEngine(engine="piper", backend=FakeBackend(fail=True))
    tts.speak_blocking("Should not hang")
    tts.stop()


def test_pyttsx3_backend_selection():
    tts = TTSEngine(engine="pyttsx3", backend=FakeBackend())
    assert tts.engine_name == "pyttsx3"
    tts.stop()


def test_stop_drains_queue():
    backend = FakeBackend()
    tts = TTSEngine(engine="piper", backend=backend)
    tts.speak("one")
    tts.speak("two")
    tts.stop()
    # Queue must be empty after stop (best-effort drain).
    assert tts._queue.empty()


def test_resolve_voice_path_returns_str_or_none():
    path = _resolve_voice_path()
    assert path is None or isinstance(path, str)


# ── Issue 6: interruptible TTS ────────────────────────────────

class BlockingBackend(FakeBackend):
    """play() blocks until interrupted — simulates long speech."""

    def __init__(self):
        super().__init__()
        self.play_started = threading.Event()
        self.release = threading.Event()
        self.play_calls = 0
        self.interrupted = False
        self._active = 0
        self.max_active = 0

    def play(self, audio):
        self.play_calls += 1
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        self.play_started.set()
        self.release.wait(timeout=5)
        self._active -= 1

    def interrupt(self):
        self.interrupted = True
        self.release.set()


def test_stop_interrupts_active_playback():
    """Speech starts -> stop() -> the current utterance is interrupted
    and the interrupt flag reaches the backend."""
    backend = BlockingBackend()
    tts = TTSEngine(engine="piper", backend=backend)
    tts.speak("long sentence")
    assert backend.play_started.wait(timeout=2), "speech never started"

    tts.stop()
    assert backend.interrupted is True
    assert tts._queue.empty()
    tts.stop()


def test_second_response_cancels_previous_cleanly():
    """A second speak() while the first is playing cancels the first
    (via stop) instead of overlapping. The queue serializes playback:
    at most one utterance is ever playing at a time."""
    backend = BlockingBackend()
    tts = TTSEngine(engine="piper", backend=backend)

    tts.speak("first response")
    assert backend.play_started.wait(timeout=2)

    # Interrupt before queueing the second response.
    tts.stop()
    tts.speak("second response")

    # Both utterances were handled, but never at the same time.
    backend.release.wait(timeout=2)
    tts.wait()
    tts.stop()
    assert backend.interrupted is True   # first was cancelled, not finished
    assert backend.play_calls == 2       # first + second
    assert backend.max_active == 1       # never overlapped


def test_stale_queued_item_skipped_after_stop():
    """Items queued before a stop() are never spoken afterwards."""
    backend = FakeBackend()
    tts = TTSEngine(engine="piper", backend=backend)
    tts.speak("one")
    tts.wait()
    tts.speak("two")
    tts.stop()
    tts.speak("three")
    tts.wait()
    tts.stop()
    # "two" was cancelled; only "one" and "three" played.
    assert backend.spoken == ["played", "played"]


def test_speak_blocking_unblocked_by_stop():
    """A blocking speaker must not hang forever when stop() is called."""
    backend = BlockingBackend()
    tts = TTSEngine(engine="piper", backend=backend)

    result = {}

    def _blocking_call():
        result["done"] = False
        tts.speak_blocking("farewell")
        result["done"] = True

    thread = threading.Thread(target=_blocking_call)
    thread.start()
    assert backend.play_started.wait(timeout=2)

    tts.stop()
    thread.join(timeout=3)
    assert thread.is_alive() is False  # unblocked
    assert result["done"] is True


def test_generation_increments_on_stop():
    tts = TTSEngine(engine="piper", backend=FakeBackend())
    gen0 = tts._current_generation()
    tts.stop()
    assert tts._current_generation() == gen0 + 1
    tts.stop()
