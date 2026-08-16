"""Doctor tests — must never crash, must report pass/fail correctly."""


import sys

import requests

import jarvis_cli.doctor as doctor


def test_safe_catches_exceptions():
    def boom():
        raise RuntimeError("kaboom")

    ok, detail = doctor._safe(boom)
    assert ok is False
    assert "kaboom" in detail


def test_safe_returns_result():
    ok, detail = doctor._safe(lambda: (True, "fine"))
    assert ok is True
    assert detail == "fine"


def test_run_doctor_returns_int(capsys):
    # Uses the real environment; every check is defensive, so this
    # must never raise — even with no mic / no Ollama present.
    result = doctor.run_doctor()
    out = capsys.readouterr().out
    assert result in (0, 1)
    assert "JARVIS DOCTOR" in out
    assert "checks passed" in out or "check(s) failed" in out


def test_run_doctor_reports_failure(monkeypatch, capsys):
    def boom():
        raise RuntimeError("mic gone")

    # Replace only the microphone check with a guaranteed failure.
    checks = []
    for name, fn, fix in doctor.CHECKS:
        if name == "Microphone":
            checks.append((name, boom, fix))
        else:
            checks.append((name, fn, fix))
    monkeypatch.setattr(doctor, "CHECKS", checks)

    result = doctor.run_doctor()
    out = capsys.readouterr().out
    assert result == 1
    assert "[✗]" in out
    assert "Fix:" in out


def test_run_doctor_reports_all_pass(monkeypatch, capsys):
    def ok_check():
        return True, "fine"

    checks = [(name, ok_check, fix) for name, _, fix in doctor.CHECKS]
    monkeypatch.setattr(doctor, "CHECKS", checks)

    result = doctor.run_doctor()
    out = capsys.readouterr().out
    assert result == 0
    assert "[✓]" in out


def test_console_marks_unicode_when_supported(monkeypatch):
    class Utf8Stdout:
        encoding = "utf-8"

    monkeypatch.setattr(sys, "stdout", Utf8Stdout())
    ok, fail, warn = doctor._console_marks()
    assert ok == "[✓]"
    assert fail == "[✗]"
    assert warn == "[!]"


def test_console_marks_ascii_fallback_for_cp1252(monkeypatch):
    """The doctor must never crash when output is piped under the
    Windows cp1252 code page (Unicode ✓/✗ cannot be encoded there)."""
    class Cp1252Stdout:
        encoding = "cp1252"

    monkeypatch.setattr(sys, "stdout", Cp1252Stdout())
    ok, fail, warn = doctor._console_marks()
    assert ok == "[OK]"
    assert fail == "[FAIL]"
    assert warn == "[WARN]"


def test_console_marks_handles_missing_encoding(monkeypatch):
    class NoEncodingStdout:
        encoding = None

    monkeypatch.setattr(sys, "stdout", NoEncodingStdout())
    ok, _, _ = doctor._console_marks()
    assert ok in ("[✓]", "[OK]")


def test_run_doctor_ascii_marks_no_crash(monkeypatch, capsys):
    """End-to-end: under a cp1252 stdout the doctor must still print
    every line (with [OK]/[FAIL]/[WARN]) instead of raising a
    UnicodeEncodeError."""
    class Cp1252Stdout:
        encoding = "cp1252"

        def __init__(self, real):
            self._real = real

        def write(self, s):
            self._real.write(s)

        def flush(self):
            self._real.flush()

    monkeypatch.setattr(sys, "stdout", Cp1252Stdout(sys.stdout))
    result = doctor.run_doctor()
    out = capsys.readouterr().out
    assert result in (0, 1)
    assert "[OK]" in out or "[FAIL]" in out
    assert "JARVIS DOCTOR" in out


def test_check_ollama_streaming(monkeypatch):
    from config import ollama_config

    monkeypatch.setattr(ollama_config, "STREAM", True)
    ok, detail = doctor._check_ollama_streaming()
    assert ok is True and "enabled" in detail

    monkeypatch.setattr(ollama_config, "STREAM", False)
    ok, detail, status = doctor._check_ollama_streaming()
    assert ok is True and status == "warn"


def test_check_thinking(monkeypatch):
    from config import ollama_config

    monkeypatch.setattr(ollama_config, "THINK", False)
    ok, detail = doctor._check_thinking()
    assert ok is True and "disabled" in detail

    monkeypatch.setattr(ollama_config, "THINK", True)
    ok, detail, status = doctor._check_thinking()
    assert ok is True and status == "warn"


def test_check_model_keepalive_loaded(monkeypatch):
    from config import ollama_config
    import requests

    monkeypatch.setattr(ollama_config, "KEEP_ALIVE", "30m")
    monkeypatch.setattr(ollama_config, "MODEL", "qwen3:8b")
    monkeypatch.setattr(
        ollama_config, "BASE_URL", "http://fake:11434"
    )

    class FakeResp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "qwen3:8b"}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    ok, detail = doctor._check_model_keepalive()
    assert ok is True
    assert "loaded" in detail
    assert "30m" in detail


def test_check_model_keepalive_not_loaded_warns(monkeypatch):
    from config import ollama_config

    monkeypatch.setattr(ollama_config, "KEEP_ALIVE", "30m")
    monkeypatch.setattr(ollama_config, "MODEL", "qwen3:8b")
    monkeypatch.setattr(
        ollama_config, "BASE_URL", "http://fake:11434"
    )

    class FakeResp:
        status_code = 200

        def json(self):
            return {"models": []}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    ok, detail, status = doctor._check_model_keepalive()
    assert ok is True
    assert status == "warn"


def test_check_model_keepalive_no_keepalive_warns(monkeypatch):
    from config import ollama_config

    monkeypatch.setattr(ollama_config, "KEEP_ALIVE", "")
    ok, detail, status = doctor._check_model_keepalive()
    assert ok is True
    assert status == "warn"


# ── Issue 7: Piper diagnostics ────────────────────────────────

def test_check_piper_installed(monkeypatch):
    """Valid Piper install -> healthy with the version reported."""
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: "1.6.1"
    )
    monkeypatch.setattr("importlib.metadata.PackageNotFoundError", PackageNotFoundError)
    monkeypatch.setattr(doctor, "os_name", lambda: "nt")
    monkeypatch.setattr("shutil.which", lambda exe: None)
    ok, detail = doctor._check_piper()
    assert ok is True
    assert "1.6.1" in detail


def test_check_piper_missing_package(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def no_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", no_version)
    monkeypatch.setattr("importlib.metadata.PackageNotFoundError", PackageNotFoundError)
    ok, detail = doctor._check_piper()
    assert ok is False
    assert "not installed" in detail


def test_check_piper_finds_executable(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: "1.6.1"
    )
    monkeypatch.setattr("importlib.metadata.PackageNotFoundError", PackageNotFoundError)
    monkeypatch.setattr(doctor, "os_name", lambda: "nt")
    monkeypatch.setattr(
        "shutil.which", lambda exe: r"C:\tools\piper.exe" if exe == "piper.exe" else None
    )
    ok, detail = doctor._check_piper()
    assert ok is True
    assert "piper.exe" in detail


def test_check_voice_model_missing_reports_path(monkeypatch, tmp_path):
    from config import tts_config

    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tts_config, "VOICE", "en_US-fancy-mega")
    monkeypatch.setattr(tts_config, "VOICE_PATH", "")
    ok, detail = doctor._check_voice_model()
    assert ok is False
    assert "en_US-fancy-mega" in detail
    assert "download" in detail  # actionable fix hint


def test_check_voice_model_present(monkeypatch, tmp_path):
    from config import tts_config

    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "en_US-lessac-medium.onnx").write_bytes(b"x")
    (tmp_path / "voices" / "en_US-lessac-medium.onnx.json").write_bytes(b"{}")
    monkeypatch.setattr(doctor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tts_config, "VOICE", "en_US-lessac-medium")
    monkeypatch.setattr(tts_config, "VOICE_PATH", "")
    ok, detail = doctor._check_voice_model()
    assert ok is True
    assert "onnx" in detail


def test_check_voice_model_configured_path_missing(monkeypatch, tmp_path):
    from config import tts_config

    monkeypatch.setattr(tts_config, "VOICE_PATH", str(tmp_path / "nope.onnx"))
    ok, detail = doctor._check_voice_model()
    assert ok is False
    assert "TTS_VOICE_PATH" in detail
    assert "nope.onnx" in detail


def test_check_config_reports_validation_problems(monkeypatch):
    from config import tts_config

    monkeypatch.setattr(tts_config, "ENGINE", "gcloud")
    ok, detail = doctor._check_config()
    assert ok is False
    assert "TTS_ENGINE" in detail
    assert "gcloud" in detail
