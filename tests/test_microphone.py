"""Microphone manager tests — sounddevice mocked."""

import engine.microphone as mic_mod
from engine.microphone import MicrophoneManager


class FakeDeviceInfo:
    def get(self, key, default=None):
        return {"name": "Fake Mic Array"}.get(key, default)


def _patch_sd(monkeypatch, default_input, query_ok=True):
    class FakeSD:
        @staticmethod
        def query_devices(device=None):
            if not query_ok:
                raise RuntimeError("PortAudio error")
            if device is None:
                return [{"name": "out"}]
            return FakeDeviceInfo()

    monkeypatch.setattr(mic_mod.sd, "query_devices", FakeSD.query_devices)
    monkeypatch.setattr(mic_mod.sd, "default", type("D", (), {"device": [default_input, None]})())


def test_detects_microphone(monkeypatch):
    _patch_sd(monkeypatch, default_input=1)
    mic = MicrophoneManager()
    assert mic.is_available() is True
    assert mic.device_name == "Fake Mic Array"
    assert "OK" in mic.describe()


def test_no_default_input(monkeypatch):
    _patch_sd(monkeypatch, default_input=None)
    mic = MicrophoneManager()
    assert mic.is_available() is False
    assert "UNAVAILABLE" in mic.describe()


def test_query_error_reports_unavailable(monkeypatch):
    _patch_sd(monkeypatch, default_input=1, query_ok=False)
    mic = MicrophoneManager()
    assert mic.is_available() is False


def test_explicit_input_device(monkeypatch):
    _patch_sd(monkeypatch, default_input=None)
    mic = MicrophoneManager(input_device=2)
    assert mic.is_available() is True
    assert mic.device_name == "Fake Mic Array"
