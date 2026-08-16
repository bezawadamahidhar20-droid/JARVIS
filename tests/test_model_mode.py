"""Fast/quality model-mode tests — model selection from .env settings.

Exercises OllamaConfig.resolve_model() (JARVIS_MODEL_MODE with
OLLAMA_FAST_MODEL / OLLAMA_QUALITY_MODEL), config validation, and the
OllamaClient default-model wiring.
"""

from config import (
    jarvis_config,
    ollama_config,
    validate_config,
)


def _setup(monkeypatch, model="qwen3:8b", fast="", quality="", mode="quality"):
    monkeypatch.setattr(ollama_config, "MODEL", model)
    monkeypatch.setattr(ollama_config, "FAST_MODEL", fast)
    monkeypatch.setattr(ollama_config, "QUALITY_MODEL", quality)
    monkeypatch.setattr(jarvis_config, "MODEL_MODE", mode)


# ── resolve_model() ───────────────────────────────────────────

def test_resolve_default_keeps_configured_model(monkeypatch):
    """Default mode + empty *_MODEL vars = current behavior (OLLAMA_MODEL)."""
    _setup(monkeypatch, model="qwen3:8b")
    assert ollama_config.resolve_model() == "qwen3:8b"


def test_resolve_fast_mode_uses_fast_model(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", fast="qwen3:1.7b", mode="fast")
    assert ollama_config.resolve_model() == "qwen3:1.7b"


def test_resolve_quality_mode_uses_quality_model(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", quality="llama3.2:3b", mode="quality")
    assert ollama_config.resolve_model() == "llama3.2:3b"


def test_resolve_fast_without_fast_model_falls_back(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", fast="", mode="fast")
    assert ollama_config.resolve_model() == "qwen3:8b"


def test_resolve_unknown_mode_falls_back_to_model(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", fast="qwen3:1.7b", mode="banana")
    assert ollama_config.resolve_model() == "qwen3:8b"


def test_resolve_explicit_mode_overrides_config(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", fast="qwen3:1.7b", mode="quality")
    assert ollama_config.resolve_model(mode="fast") == "qwen3:1.7b"


def test_resolve_strips_whitespace(monkeypatch):
    _setup(monkeypatch, model=" qwen3:8b ", fast=" qwen3:1.7b ", mode="fast")
    assert ollama_config.resolve_model() == "qwen3:1.7b"


# ── OllamaClient default-model wiring ─────────────────────────

def test_ollama_client_default_uses_resolved_model(monkeypatch):
    import brain.ollama_client as oc

    _setup(monkeypatch, model="qwen3:8b", fast="qwen3:1.7b", mode="fast")
    # The client resolves the default lazily from the config singleton.
    monkeypatch.setattr(oc.requests, "get", lambda *a, **k: _FakeTags())
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: _FakeTags())
    client = oc.OllamaClient(base_url="http://fake:11434")
    assert client.model == "qwen3:1.7b"
    assert "qwen3:1.7b" in client.describe()


def test_ollama_client_explicit_model_wins(monkeypatch):
    import brain.ollama_client as oc

    _setup(monkeypatch, model="qwen3:8b", fast="qwen3:1.7b", mode="fast")
    monkeypatch.setattr(oc.requests, "get", lambda *a, **k: _FakeTags())
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: _FakeTags())
    client = oc.OllamaClient(base_url="http://fake:11434", model="llama3.2:3b")
    assert client.model == "llama3.2:3b"


class _FakeTags:
    status_code = 200

    def json(self):
        return {"models": [{"name": "qwen3:8b"}, {"name": "qwen3:1.7b"}]}

    def raise_for_status(self):
        pass


# ── Validation ────────────────────────────────────────────────

def test_validate_accepts_fast_and_quality_modes(monkeypatch):
    for mode in ("fast", "quality"):
        _setup(monkeypatch, fast="qwen3:1.7b", quality="qwen3:8b", mode=mode)
        problems = validate_config()
        matching = [p for p in problems if p["setting"] == "JARVIS_MODEL_MODE"]
        assert matching == [], mode


def test_validate_rejects_unknown_mode(monkeypatch):
    _setup(monkeypatch, mode="turbo")
    problems = validate_config()
    matching = [p for p in problems if p["setting"] == "JARVIS_MODEL_MODE"]
    assert matching and matching[0]["fatal"] is False


def test_validate_warns_when_fast_mode_has_no_fast_model(monkeypatch):
    _setup(monkeypatch, model="qwen3:8b", fast="", mode="fast")
    problems = validate_config()
    matching = [p for p in problems if p["setting"] == "OLLAMA_FAST_MODEL"]
    assert matching and matching[0]["fatal"] is False
