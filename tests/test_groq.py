"""Issue 8 — Groq provider (all HTTP mocked, no key required).

Groq is optional and must degrade gracefully with no API key.
"""

import brain.groq_client as gc
from brain.groq_client import GroqClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, lines=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.text = text

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


def _client(api_key="test-key"):
    return GroqClient(api_key=api_key, model="llama-3.3-70b-versatile")


# ── Configuration / availability ──────────────────────────────

def test_unconfigured_is_unavailable():
    client = _client(api_key="")
    assert client.is_configured() is False
    assert client.is_available() is False


def test_configured_is_available():
    client = _client()
    assert client.is_configured() is True
    assert client.is_available() is True


def test_no_key_ask_returns_none_without_request(monkeypatch):
    called = []

    def fake_post(*a, **k):
        called.append(a)
        return FakeResponse(200, {"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setattr(gc.requests, "post", fake_post)
    assert _client(api_key="").ask("hello") is None
    assert called == []  # never hit the network


def test_no_key_stream_returns_none(monkeypatch):
    monkeypatch.setattr(gc.requests, "post", lambda *a, **k: FakeResponse())
    assert _client(api_key="").ask_stream("hello") is None


# ── ask / ask_stream ──────────────────────────────────────────

def test_ask_success(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.update(headers=headers, json=json)
        return FakeResponse(200, {
            "choices": [{"message": {"content": "**Yes**, the sky is blue."}}],
        })

    monkeypatch.setattr(gc.requests, "post", fake_post)
    result = _client().ask("is the sky blue")
    assert result == "Yes, the sky is blue."
    # Key travels in the Authorization header — never in the body/logs.
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["json"]["model"] == "llama-3.3-70b-versatile"
    assert calls["json"]["stream"] is False


def test_ask_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(gc.requests, "post",
                        lambda *a, **k: FakeResponse(401, {"error": "bad key"}))
    assert _client().ask("hello") is None


def test_ask_network_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise gc.requests.ConnectionError("down")

    monkeypatch.setattr(gc.requests, "post", boom)
    assert _client().ask("hello") is None


def test_ask_stream_emits_sentences(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"First sentence."}}]}',
        'data: {"choices":[{"delta":{"content":" Second sentence!"}}]}',
        "data: [DONE]",
    ]

    class StreamResp(FakeResponse):
        def iter_lines(self, decode_unicode=False):
            if decode_unicode:
                return iter(lines)
            return iter(line.encode() for line in lines)

    monkeypatch.setattr(gc.requests, "post", lambda *a, **k: StreamResp(200))
    spoken = []
    result = _client().ask_stream("hi", memory=None, on_sentence=spoken.append)
    assert result == "First sentence. Second sentence!"
    assert spoken == ["First sentence.", "Second sentence!"]


# ── Factory wiring ────────────────────────────────────────────

def test_create_provider_groq_when_configured(monkeypatch):
    from brain.llm import create_provider
    from config import jarvis_config

    monkeypatch.setattr(jarvis_config, "AI_PROVIDER", "groq")
    monkeypatch.setattr(gc, "GROQ_API_KEY", "key-123")
    provider = create_provider()
    assert provider.name == "groq"
    assert provider.is_configured()


def test_create_provider_ollama_by_default(monkeypatch):
    from brain.llm import create_provider
    from config import jarvis_config

    monkeypatch.setattr(jarvis_config, "AI_PROVIDER", "ollama")
    monkeypatch.setattr(
        "brain.ollama_client.requests.get",
        lambda *a, **k: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen3:8b"}]},
        })(),
    )
    provider = create_provider()
    assert provider.name == "ollama"

