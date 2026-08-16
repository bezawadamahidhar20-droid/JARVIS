"""Adaptive VAD tests — synthetic audio, no hardware required."""

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
    class BoomStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            raise vad.sd.PortAudioError("device disconnected")

    monkeypatch.setattr(vad.sd, "InputStream", lambda **k: BoomStream())
    v = AdaptiveVAD()
    assert v.record_phrase() is None
