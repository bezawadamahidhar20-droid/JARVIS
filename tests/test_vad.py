"""Adaptive VAD tests — synthetic audio, no hardware required.

record_phrase() is event-driven (callback-based InputStream): the tests
inject a fake stream that invokes the callback from a worker thread,
exactly like PortAudio does.
"""

import threading

import numpy as np

import engine.vad as vad
from engine.vad import AdaptiveVAD, rms, derive_threshold


def test_rms_silence_is_zero():
    chunk = np.zeros(800, dtype=np.int16)
    assert rms(chunk) == 0.0


def test_rms_of_loud_signal():
    chunk = np.full(800, 10000, dtype=np.int16)
    assert rms(chunk) == 10000.0


def test_rms_empty_chunk():
    assert rms(np.array([], dtype=np.int16)) == 0.0
    assert rms(None) == 0.0


def test_derive_threshold_quiet_room_floors():
    assert derive_threshold(0.0) == vad.MIN_THRESHOLD
    assert derive_threshold(10.0) == vad.MIN_THRESHOLD


def test_derive_threshold_noisy_room_scales():
    ambient = 500.0
    expected = max(vad.MIN_THRESHOLD, ambient * vad.THRESHOLD_MULTIPLIER)
    assert derive_threshold(ambient) == expected


def test_calibrate_handles_stream_failure(monkeypatch):
    class Boom:
        def __enter__(self):
            raise vad.sd.PortAudioError("no device")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(vad.sd, "InputStream", Boom)
    v = AdaptiveVAD()
    v.threshold = 999.0
    v.calibrate()  # must not raise
    assert v.calibrated is False
    assert v.threshold == 999.0  # unchanged


def test_calibrate_disabled_when_seconds_zero(monkeypatch):
    monkeypatch.setattr(vad, "CALIBRATE_SECONDS", 0.0)
    v = AdaptiveVAD()
    v.calibrate()
    assert v.calibrated is False


def test_record_phrase_stream_error_returns_none(monkeypatch):
    """A dead device must fail fast (return None), never hang."""
    class Boom:
        def __enter__(self):
            raise vad.sd.PortAudioError("no device")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(vad.sd, "InputStream", lambda **k: Boom())
    v = AdaptiveVAD()
    assert v.record_phrase() is None


def _fake_stream_fixture(monkeypatch):
    """Install a fake InputStream that records the callback and signals
    when the stream is open (so tests can drive the callback)."""
    captured = {}
    stream_ready = threading.Event()

    class FakeStream:
        def __init__(self, **kwargs):
            captured["callback"] = kwargs.get("callback")

        def __enter__(self):
            stream_ready.set()
            return self

        def __exit__(self, *a):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(vad.sd, "InputStream", lambda **k: FakeStream(**k))
    return captured, stream_ready


def test_record_phrase_detects_speech_via_callback(monkeypatch):
    """Issue 4: event-driven capture — the callback feeds chunks and
    signals completion; loud speech followed by silence returns audio."""
    captured, stream_ready = _fake_stream_fixture(monkeypatch)
    monkeypatch.setattr(vad, "VERBOSE", False)
    monkeypatch.setattr(vad, "TIMEOUT", 5)
    monkeypatch.setattr(vad, "CHUNK_DURATION", 0.05)
    monkeypatch.setattr(vad, "SILENCE_DURATION", 0.7)

    v = AdaptiveVAD()
    v.threshold = 100.0
    frame = v.sample_rate // 20  # 0.05s @ 16 kHz

    result = {}
    thread = threading.Thread(
        target=lambda: result.update(audio=v.record_phrase())
    )
    thread.start()
    assert stream_ready.wait(timeout=2), "stream never opened"

    # Drive the callback exactly like PortAudio would.
    cb = captured["callback"]
    loud = np.full((frame, 1), 8000, dtype=np.int16)
    quiet = np.zeros((frame, 1), dtype=np.int16)
    for _ in range(4):
        cb(loud, 0, None, None)
    for _ in range(20):  # silence -> phrase ends
        cb(quiet, 0, None, None)

    thread.join(timeout=3)
    assert not thread.is_alive()
    audio = result.get("audio")
    assert audio is not None
    assert audio.size > 0
    assert audio.dtype == np.int16


def test_record_phrase_timeout_when_no_speech(monkeypatch):
    """Silence continues -> the no-speech timeout fires and
    record_phrase gives up with None instead of spinning."""
    captured, stream_ready = _fake_stream_fixture(monkeypatch)
    monkeypatch.setattr(vad, "VERBOSE", False)
    monkeypatch.setattr(vad, "TIMEOUT", 1)  # max_wait = 20 chunks
    monkeypatch.setattr(vad, "CHUNK_DURATION", 0.05)

    v = AdaptiveVAD()
    v.threshold = 100.0
    frame = v.sample_rate // 20

    result = {}
    thread = threading.Thread(
        target=lambda: result.update(audio=v.record_phrase())
    )
    thread.start()
    assert stream_ready.wait(timeout=2), "stream never opened"

    cb = captured["callback"]
    quiet = np.zeros((frame, 1), dtype=np.int16)
    for _ in range(30):  # silence the whole time -> timeout
        cb(quiet, 0, None, None)

    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result.get("audio") is None
    assert v.live_speech is False
