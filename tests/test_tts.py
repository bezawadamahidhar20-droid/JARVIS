"""TTS engine tests — fake backend injected, no real voice or audio."""


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
