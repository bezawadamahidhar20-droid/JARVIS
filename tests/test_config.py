"""Configuration tests — typed getters and safe fallbacks.

The config module is imported once with override=True (the .env file is
authoritative), so these tests exercise the pure getter helpers with
monkeypatched env vars instead of reloading the module.
"""

from config import (
    _env_bool,
    _env_float,
    _env_int,
    _env_str,
    ollama_config,
    stt_config,
    tts_config,
    vad_config,
    whisper_config,
    jarvis_config,
)


def test_env_str(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_STR", "hello")
    assert _env_str("JARVIS_TEST_STR", "dflt") == "hello"
    monkeypatch.delenv("JARVIS_TEST_STR")
    assert _env_str("JARVIS_TEST_STR", "dflt") == "dflt"


def test_env_int(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_INT", "42")
    assert _env_int("JARVIS_TEST_INT", 1) == 42
    monkeypatch.setenv("JARVIS_TEST_INT", "junk")
    assert _env_int("JARVIS_TEST_INT", 1) == 1  # fallback
    monkeypatch.delenv("JARVIS_TEST_INT")
    assert _env_int("JARVIS_TEST_INT", 1) == 1


def test_env_float(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_FLOAT", "0.75")
    assert _env_float("JARVIS_TEST_FLOAT", 0.5) == 0.75
    monkeypatch.setenv("JARVIS_TEST_FLOAT", "abc")
    assert _env_float("JARVIS_TEST_FLOAT", 0.5) == 0.5


def test_env_bool(monkeypatch):
    for raw, expected in (
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("garbage", True),  # unknown -> default
        ("", True),         # empty -> default
    ):
        monkeypatch.setenv("JARVIS_TEST_BOOL", raw)
        assert _env_bool("JARVIS_TEST_BOOL", True) == expected, raw


def test_loaded_singletons_have_working_defaults():
    # These reflect the real .env on this machine; assert shape, not
    # exact values, so the suite is robust to .env edits.
    assert ollama_config.BASE_URL.startswith("http")
    assert ollama_config.MODEL
    assert 0 <= ollama_config.TEMPERATURE <= 1
    assert whisper_config.MODEL
    assert whisper_config.COMPUTE_TYPE in ("int8", "float16", "int8_float16")
    assert stt_config.SAMPLE_RATE > 0
    assert tts_config.ENGINE in ("piper", "pyttsx3")
    assert vad_config.SILENCE_DURATION > 0
    assert jarvis_config.AI_PROVIDER


def test_singletons_are_consistent():
    # The getters must never return the "missing" sentinels.
    assert isinstance(ollama_config.MODEL, str) and len(ollama_config.MODEL) > 0
    assert isinstance(whisper_config.MODEL, str) and len(whisper_config.MODEL) > 0
    assert stt_config.INPUT_DEVICE is None or stt_config.INPUT_DEVICE >= 0
