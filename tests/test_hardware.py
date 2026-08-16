"""Hardware report tests — `jarvis --hardware` with all probes mocked.

The report must be read-only, never crash on missing data, and label
CPU-only machines clearly.
"""

import jarvis_cli.hardware as hw


def _run(monkeypatch, capsys, cpu="Intel(R) Core(TM) i7", cores=8,
         ram=16.0, gpus=(), version="0.5.4", size_gb=4.7):
    monkeypatch.setattr(hw, "_cpu_name", lambda: cpu)
    monkeypatch.setattr(hw.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(hw, "_ram_gb", lambda: ram)
    monkeypatch.setattr(hw, "_gpu_names", lambda: list(gpus))
    monkeypatch.setattr(hw, "_ollama_version", lambda base: version)
    monkeypatch.setattr(hw, "_model_size_gb", lambda base, m: size_gb)
    return hw.run_hardware()


def test_hardware_report_cpu_only(monkeypatch, capsys):
    code = _run(monkeypatch, capsys, gpus=())
    assert code == 0
    out = capsys.readouterr().out
    assert "HARDWARE REPORT" in out
    assert "Intel(R) Core(TM) i7" in out
    assert "8 logical cores" in out
    assert "16.0 GB total" in out
    assert "CPU-only mode" in out
    assert "read-only" in out.lower()
    assert "no hardware or configuration changes" in out


def test_hardware_report_gpu_detected(monkeypatch, capsys):
    _run(monkeypatch, capsys, gpus=("NVIDIA GeForce RTX 3060",))
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 3060" in out
    assert "CPU-only mode" not in out
    assert "GPU acceleration available" in out


def test_hardware_report_integrated_graphics_is_cpu_only(monkeypatch, capsys):
    """An Intel/Radeon integrated GPU is listed by Windows but cannot
    accelerate Ollama — the report must still say CPU-only mode."""
    _run(monkeypatch, capsys, gpus=("Intel(R) Graphics",))
    out = capsys.readouterr().out
    assert "Intel(R) Graphics" in out
    assert "CPU-only mode detected" in out
    assert "GPU acceleration available" not in out


def test_hardware_model_size_exact_name_match(monkeypatch):
    """The size lookup must use the exact model name — qwen3:8b must
    not report qwen3:1.7b's size when qwen3:1.7b is listed first."""
    import requests

    class Tags:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [
                {"name": "qwen3:1.7b", "size": int(1.4e9)},  # listed first
                {"name": "qwen3:8b", "size": int(5.2e9)},
            ]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: Tags())
    gb = hw._model_size_gb("http://fake", "qwen3:8b")
    assert gb is not None
    assert 4.5 < gb < 5.5  # ~5.2 GB (GiB), NOT the 1.4 GB of qwen3:1.7b


def test_hardware_report_ollama_version_and_model(monkeypatch, capsys):
    _run(monkeypatch, capsys)
    out = capsys.readouterr().out
    assert "0.5.4" in out
    # Resolved model name + size appear (whatever .env selects).
    from config import ollama_config

    assert ollama_config.resolve_model() in out
    assert "4.7 GB" in out


def test_hardware_report_ollama_down(monkeypatch, capsys):
    _run(monkeypatch, capsys, version=None, size_gb=None)
    out = capsys.readouterr().out
    assert "not reachable" in out
    assert "unknown" in out


def test_hardware_report_unknown_values_do_not_crash(monkeypatch, capsys):
    monkeypatch.setattr(hw, "_cpu_name", lambda: "unknown")
    monkeypatch.setattr(hw, "_ram_gb", lambda: None)
    monkeypatch.setattr(hw, "_gpu_names", lambda: [])
    monkeypatch.setattr(hw, "_ollama_version", lambda base: None)
    monkeypatch.setattr(hw, "_model_size_gb", lambda base, m: None)
    assert hw.run_hardware() == 0
    out = capsys.readouterr().out
    assert "CPU-only mode" in out
    assert "unknown" in out
