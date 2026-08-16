"""AI provider factory tests."""

import pytest

from brain.llm import create_provider, LLMProvider
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
