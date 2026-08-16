"""Issue 8 — Groq fallback provider (all HTTP mocked, no key required).

Ollama stays the primary provider; Groq is optional and must degrade
gracefully with no API key.
"""

import brain.groq_client as gc
from brain.groq_client import GroqClient
from brain.llm import FallbackProvider, create_provider


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


# ── FallbackProvider ──────────────────────────────────────────

class FakePrimary:
    name = "ollama"

    def __init__(self, available=True):
        self.available = available
        self.asks = 0

    def is_available(self):
        return self.available

    def ask(self, user_input, memory=None, context=None):
        self.asks += 1
        if not self.available:
            return None
        return "primary answer"

    def ask_stream(self, user_input, memory=None, on_sentence=None, context=None):
        if not self.available:
            return None
        if on_sentence:
            on_sentence("primary answer")
        return "primary answer"

    def describe(self):
        return "ollama (primary)"

    def warmup(self):
        pass


class FakeGroq:
    name = "groq"

    def __init__(self):
        self.asks = 0

    def is_available(self):
        return True

    def ask(self, user_input, memory=None, context=None):
        self.asks += 1
        return "groq fallback answer"

    def ask_stream(self, user_input, memory=None, on_sentence=None, context=None):
        self.asks += 1
        if on_sentence:
            on_sentence("groq fallback answer")
        return "groq fallback answer"

    def describe(self):
        return "groq"


def test_primary_used_when_available():
    primary, fallback = FakePrimary(available=True), FakeGroq()
    provider = FallbackProvider(primary, fallback)
    assert provider.ask("hi") == "primary answer"
    assert provider.ask_stream("hi") == "primary answer"
    assert fallback.asks == 0


def test_fallback_used_when_primary_down():
    primary, fallback = FakePrimary(available=False), FakeGroq()
    provider = FallbackProvider(primary, fallback)
    assert provider.ask("hi") == "groq fallback answer"
    assert provider.ask_stream("hi") == "groq fallback answer"
    assert fallback.asks == 2


def test_fallback_is_available_when_primary_down():
    provider = FallbackProvider(FakePrimary(available=False), FakeGroq())
    assert provider.is_available() is True


def test_no_fallback_when_none_given():
    primary = FakePrimary(available=False)
    provider = FallbackProvider(primary, None)
    assert provider.ask("hi") is None
    assert provider.is_available() is False


# ── Factory wiring ────────────────────────────────────────────

def test_create_provider_ollama_with_groq_key_wraps_fallback(monkeypatch):
    from config import groq_config

    monkeypatch.setattr(groq_config, "API_KEY", "key-123")
    monkeypatch.setattr(
        "brain.ollama_client.requests.get",
        lambda *a, **k: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen3:8b"}]},
        })(),
    )
    provider = create_provider("ollama")
    assert isinstance(provider, FallbackProvider)
    assert provider.primary.name == "ollama"
    assert provider.fallback is not None


def test_create_provider_ollama_without_key_is_plain(monkeypatch):
    monkeypatch.setattr(gc, "GROQ_API_KEY", "")
    monkeypatch.setattr(
        "brain.ollama_client.requests.get",
        lambda *a, **k: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen3:8b"}]},
        })(),
    )
    provider = create_provider("ollama")
    assert not isinstance(provider, FallbackProvider)
    assert provider.name == "ollama"


def test_create_provider_groq_explicit(monkeypatch):
    monkeypatch.setattr(gc, "GROQ_API_KEY", "key-123")
    provider = create_provider("groq")
    assert provider.name == "groq"
    assert provider.is_configured()
