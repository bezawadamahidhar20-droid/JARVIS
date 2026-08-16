"""AI provider factory tests."""

import pytest

from brain.llm import FallbackProvider, create_provider, LLMProvider
from brain.ollama_client import OllamaClient


def test_create_provider_ollama(monkeypatch):
    # Avoid the real HTTP ping in __init__ for speed.
    monkeypatch.setattr(
        "brain.ollama_client.requests.get",
        lambda *a, **k: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen3:8b"}]},
        })(),
    )
    provider = create_provider("ollama")
    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, OllamaClient)


def test_create_provider_unknown():
    with pytest.raises(ValueError):
        create_provider("nonexistent-provider")


def test_create_provider_default(monkeypatch):
    monkeypatch.setattr(
        "brain.ollama_client.requests.get",
        lambda *a, **k: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen3:8b"}]},
        })(),
    )
    provider = create_provider()
    assert provider.name == "ollama"


def test_fallback_provider_switch_model_passthrough():
    """The fallback wrapper must forward runtime model switches to its
    primary (local) provider."""
    calls = []

    class Primary:
        model = "qwen3:8b"

        def switch_model(self, model):
            calls.append(model)
            self.model = model
            return model

    fallback = FallbackProvider(Primary(), fallback=None)
    assert fallback.switch_model("qwen3:1.7b") == "qwen3:1.7b"
    assert calls == ["qwen3:1.7b"]
    assert fallback.primary.model == "qwen3:1.7b"


def test_fallback_provider_switch_model_unsupported():
    class NoSwitch:
        pass

    fallback = FallbackProvider(NoSwitch(), fallback=None)
    with pytest.raises(NotImplementedError):
        fallback.switch_model("qwen3:1.7b")
