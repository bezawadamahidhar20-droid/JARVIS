"""STT engine tests — Whisper model fully mocked."""

import sys

import numpy as np

import engine.stt as stt_mod
from engine.stt import STTEngine, _detect_device


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, *a, **k):
        pass

    def transcribe(self, samples, **kwargs):
        class Seg:
            text = "Hello World"

        class Info:
            pass

        return iter([Seg()]), Info()


class EmptyWhisperModel(FakeWhisperModel):
    def transcribe(self, samples, **kwargs):
        class Info:
            pass

        return iter([]), Info()


class BoomWhisperModel(FakeWhisperModel):
    def transcribe(self, samples, **kwargs):
        raise RuntimeError("inference failed")


def _patch_faster_whisper(monkeypatch, model_cls):
    """faster_whisper is imported lazily inside load() — patch the module."""
    fake_module = type("faster_whisper", (), {"WhisperModel": model_cls})
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def _engine_with(model_cls, monkeypatch):
    _patch_faster_whisper(monkeypatch, model_cls)
    engine = STTEngine()
    engine.load()
    return engine


def test_load_sets_model(monkeypatch):
    engine = _engine_with(FakeWhisperModel, monkeypatch)
    assert engine.model is not None
    assert engine._loaded is True


def test_load_idempotent(monkeypatch):
    count = {"n": 0}

    class Counting(FakeWhisperModel):
        def __init__(self, *a, **k):
            count["n"] += 1
            super().__init__(*a, **k)

    engine = _engine_with(Counting, monkeypatch)
    engine.load()
    engine.load()
    assert count["n"] == 1


def test_transcribe_returns_lowercased_text(monkeypatch):
    engine = _engine_with(FakeWhisperModel, monkeypatch)
    audio = np.ones(16000, dtype=np.int16) * 1000
    assert engine.transcribe(audio) == "hello world"


def test_transcribe_empty_audio_returns_none(monkeypatch):
    engine = _engine_with(FakeWhisperModel, monkeypatch)
    assert engine.transcribe(np.array([], dtype=np.int16)) is None
    assert engine.transcribe(None) is None


def test_transcribe_no_segments_returns_none(monkeypatch):
    engine = _engine_with(EmptyWhisperModel, monkeypatch)
    audio = np.ones(16000, dtype=np.int16) * 1000
    assert engine.transcribe(audio) is None


def test_transcribe_error_returns_none(monkeypatch):
    engine = _engine_with(BoomWhisperModel, monkeypatch)
    audio = np.ones(16000, dtype=np.int16) * 1000
    assert engine.transcribe(audio) is None


def test_transcribe_without_load_returns_none(monkeypatch):
    _patch_faster_whisper(monkeypatch, FakeWhisperModel)
    engine = STTEngine()
    audio = np.ones(1600, dtype=np.int16)
    assert engine.transcribe(audio) is None


def test_load_missing_package(monkeypatch):
    # A None entry in sys.modules makes `import faster_whisper` raise
    # ImportError — exactly what happens when the package is absent.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    engine = STTEngine()
    assert engine.load() is False
    assert engine.model is None


# ── Issue 9: automatic CUDA detection ─────────────────────────

def _patch_ctranslate2(monkeypatch, device_count):
    fake = type("ctranslate2", (), {
        "get_cuda_device_count": staticmethod(lambda: device_count),
    })
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)


def test_detect_device_explicit_cpu():
    assert _detect_device("cpu") == "cpu"
    assert _detect_device("cuda") == "cuda"


def test_detect_device_auto_cuda_available(monkeypatch):
    _patch_ctranslate2(monkeypatch, device_count=1)
    assert _detect_device("auto") == "cuda"


def test_detect_device_auto_falls_back_to_cpu(monkeypatch):
    _patch_ctranslate2(monkeypatch, device_count=0)
    assert _detect_device("auto") == "cpu"


def test_detect_device_auto_no_ctranslate2(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)  # import fails
    assert _detect_device("auto") == "cpu"


def test_detect_device_auto_ctranslate2_error(monkeypatch):
    class Boom:
        @staticmethod
        def get_cuda_device_count():
            raise RuntimeError("no CUDA libs")

    monkeypatch.setitem(sys.modules, "ctranslate2", Boom)
    assert _detect_device("auto") == "cpu"


def test_load_uses_cuda_when_detected(monkeypatch):
    _patch_faster_whisper(monkeypatch, FakeWhisperModel)
    calls = {}

    class Recording(FakeWhisperModel):
        def __init__(self, *a, **k):
            calls.update(k)
            super().__init__(*a, **k)

    _patch_faster_whisper(monkeypatch, Recording)
    monkeypatch.setattr(stt_mod, "_detect_device", lambda pref: "cuda")
    engine = STTEngine()
    engine.load()
    assert calls.get("device") == "cuda"
    assert calls.get("compute_type") == "float16"  # int8 -> float16 on CUDA
    assert engine.device == "cuda"


def test_load_falls_back_to_cpu_when_cuda_fails(monkeypatch):
    """CUDA load failure (missing cuDNN etc.) must retry on CPU and
    never crash JARVIS on GPU-less machines."""
    calls = []

    class FlakyWhisper:
        def __init__(self, *a, **k):
            calls.append(k)
            if k.get("device") == "cuda":
                raise RuntimeError("cuDNN not found")

    _patch_faster_whisper(monkeypatch, FlakyWhisper)
    monkeypatch.setattr(stt_mod, "_detect_device", lambda pref: "cuda")
    engine = STTEngine()
    assert engine.load() is True
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert calls[0]["device"] == "cuda"  # tried CUDA first
    assert calls[-1]["device"] == "cpu"   # then fell back
