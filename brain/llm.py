"""
brain/llm.py — Provider-agnostic AI layer

JARVIS talks to its brain through the ``LLMProvider`` interface so a
new provider (Groq, NVIDIA NIM, Gemini, ...) can be added later
without rewriting the assistant. Today the only implementation is
``OllamaClient`` in brain/ollama_client.py.

Security boundary: a provider returns *text only*. It never executes
commands — the deterministic CommandRegistry is the only thing that
touches the OS (see commands/).
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger("llm")


class LLMProvider(ABC):
    """Interface every AI backend must implement."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend is reachable right now."""

    @abstractmethod
    def ask(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Return the full response text, or None on failure.

        ``context`` is optional verified information (e.g. web search
        results) the provider should answer from instead of its own
        training knowledge.
        """

    @abstractmethod
    def ask_stream(
        self,
        user_input: str,
        memory=None,
        on_sentence: Optional[Callable[[str], None]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Stream the response, handing each finished sentence to
        ``on_sentence`` (e.g. TTS). Returns the full text or None.

        ``context`` is optional verified information (e.g. web search
        results) the provider should answer from instead of its own
        training knowledge.
        """

    def warmup(self) -> None:
        """Optional: pre-load the model so the first real request is fast."""

    def describe(self) -> str:
        """Short status string for `jarvis --doctor`."""
        return self.name


def create_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Factory: build the configured AI provider.

    ``name`` defaults to jarvis_config.AI_PROVIDER (env AI_PROVIDER).
    Raises ValueError for unknown providers — the CLI catches this and
    falls back to a provider-less run (local commands only).
    """
    if name is None:
        try:
            from config import jarvis_config

            name = jarvis_config.AI_PROVIDER
        except Exception:
            name = "ollama"

    name = (name or "ollama").strip().lower()

    if name == "ollama":
        from brain.ollama_client import OllamaClient

        return OllamaClient()

    raise ValueError(
        f"Unknown AI provider '{name}'. Supported: ollama"
    )
