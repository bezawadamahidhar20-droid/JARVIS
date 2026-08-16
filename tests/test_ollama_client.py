"""Ollama client tests — all HTTP mocked, no real server required."""

import brain.ollama_client as oc
from brain.ollama_client import OllamaClient


class FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, status_code=200, json_data=None, text="", lines=None):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self._lines = lines or []

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


def _make_client(monkeypatch, get_response=None, post_response=None):
    monkeypatch.setattr(oc.requests, "get", lambda *a, **k: get_response)
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: post_response)
    return OllamaClient(base_url="http://fake:11434", model="qwen3:8b")


def test_is_available_true(monkeypatch):
    client = _make_client(monkeypatch, get_response=FakeResponse(200, {"models": []}))
    assert client.is_available() is True


def test_is_available_false(monkeypatch):
    def boom(*a, **k):
        raise oc.requests.ConnectionError("down")

    monkeypatch.setattr(oc.requests, "get", boom)
    monkeypatch.setattr(oc.requests, "post", boom)
    client = OllamaClient(base_url="http://fake:11434", model="qwen3:8b")
    assert client.is_available() is False


def test_ask_success(monkeypatch):
    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": [{"name": "qwen3:8b"}]}),
        post_response=FakeResponse(
            200,
            {"message": {"content": "**The** capital of France is Paris."}},
        ),
    )
    result = client.ask("capital of france", memory=None)
    assert result == "The capital of France is Paris."


def test_ask_connection_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise oc.requests.ConnectionError("refused")

    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": []}),
    )
    monkeypatch.setattr(oc.requests, "post", boom)
    assert client.ask("hello") is None


def test_ask_empty_response_returns_none(monkeypatch):
    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": []}),
        post_response=FakeResponse(200, {"message": {"content": ""}}),
    )
    assert client.ask("hello") is None


def test_ask_stream_emits_sentences(monkeypatch):
    lines = [
        '{"message":{"content":"Hello "},"done":false}',
        '{"message":{"content":"there."},"done":false}',
        '{"message":{"content":" How are"},"done":false}',
        '{"message":{"content":" you?"},"done":false}',
        '{"message":{"content":""},"done":true}',
    ]
    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": [{"name": "qwen3:8b"}]}),
        post_response=FakeResponse(200, lines=lines),
    )
    spoken = []
    result = client.ask_stream("hi", memory=None, on_sentence=spoken.append)
    assert result == "Hello there. How are you?"
    assert len(spoken) == 2  # two complete sentences


def test_ask_stream_http_error_returns_none(monkeypatch):
    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": []}),
        post_response=FakeResponse(500, {"error": "boom"}),
    )
    assert client.ask_stream("hi") is None


def test_extract_sentences():
    client = OllamaClient(base_url="http://fake", model="m")
    sentences, remainder = client._extract_sentences(
        "First. Second! Third? unfinished"
    )
    assert sentences == ["First.", "Second!", "Third?"]
    assert remainder == " unfinished"  # leading space is preserved


def test_extract_sentences_splits_on_newline():
    client = OllamaClient(base_url="http://fake", model="m")
    sentences, _ = client._extract_sentences("Line one\nLine two.")
    assert sentences == ["Line one", "Line two."]


def test_build_payload_has_think_false(monkeypatch):
    client = _make_client(
        monkeypatch, get_response=FakeResponse(200, {"models": []})
    )
    payload = client._build_payload(
        [{"role": "user", "content": "hi"}], stream=False
    )
    assert payload["model"] == "qwen3:8b"
    assert payload["think"] is False
    assert payload["options"]["num_predict"] > 0
    assert payload["options"]["temperature"] >= 0


def test_switch_model_changes_active_model(monkeypatch):
    client = _make_client(
        monkeypatch, get_response=FakeResponse(200, {"models": []})
    )
    assert client.model == "qwen3:8b"
    assert client.switch_model("qwen3:1.7b") == "qwen3:1.7b"
    assert client.model == "qwen3:1.7b"
    # Switching to the same model is a no-op.
    assert client.switch_model("qwen3:1.7b") == "qwen3:1.7b"


def test_switch_model_rejects_empty_name(monkeypatch):
    import pytest

    client = _make_client(
        monkeypatch, get_response=FakeResponse(200, {"models": []})
    )
    with pytest.raises(ValueError):
        client.switch_model("")
    with pytest.raises(ValueError):
        client.switch_model("   ")


def test_payload_uses_switched_model(monkeypatch):
    client = _make_client(
        monkeypatch, get_response=FakeResponse(200, {"models": []})
    )
    client.switch_model("llama3.2:3b")
    payload = client._build_payload(
        [{"role": "user", "content": "hi"}], stream=False
    )
    assert payload["model"] == "llama3.2:3b"


def test_warmup_payload_disables_thinking(monkeypatch):
    """The warm-up request must load the model in the same state as
    real requests (think disabled) — otherwise the first question pays
    for a second context setup."""
    sent = {}

    def fake_post(url, json=None, **kwargs):
        sent.update(json=json)
        return FakeResponse(200, {"done": True})

    client = _make_client(
        monkeypatch,
        get_response=FakeResponse(200, {"models": [{"name": "qwen3:8b"}]}),
        post_response=FakeResponse(200, {"done": True}),
    )
    monkeypatch.setattr(oc.requests, "post", fake_post)
    client.warmup()
    assert sent["json"]["think"] is False
    assert sent["json"]["options"]["num_predict"] == 1
    assert sent["json"]["keep_alive"] == client.keep_alive


def test_clean_response_strips_markdown():
    client = OllamaClient(base_url="http://fake", model="m")
    cleaned = client._clean_response(
        "**bold** and `code` and # header\n\n\n\nmore text"
    )
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "header" in cleaned
