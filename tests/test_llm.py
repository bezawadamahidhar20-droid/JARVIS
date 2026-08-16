"""AI provider factory tests."""

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
    provider = create_provider()
    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, OllamaClient)


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


def test_create_provider_groq(monkeypatch):
    from config import jarvis_config

    monkeypatch.setattr(jarvis_config, "AI_PROVIDER", "groq")
    provider = create_provider()
    assert provider.name == "groq"
