"""
brain/llm.py — Provider-agnostic AI layer

JARVIS talks to its brain through the ``LLMProvider`` interface so a
new provider (Groq, NVIDIA NIM, Gemini, ...) can be added later
without rewriting the assistant. Today:

  * ``OllamaClient``  — the local, primary provider (brain/ollama_client.py)
  * ``GroqClient``    — optional cloud fallback (brain/groq_client.py)
  * ``FallbackProvider`` — tries the primary, then the fallback, so
    Ollama stays the default and Groq only steps in when Ollama fails.

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


class FallbackProvider(LLMProvider):
    """
    Wraps a primary provider (Ollama) with an optional fallback (Groq).

    Every call tries the primary first; if it fails or returns None the
    fallback is used. ``is_available()`` is True when either backend is
    usable, so JARVIS keeps conversational answers when Ollama is down.
    """

    name = "fallback"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None):
        self.primary = primary
        self.fallback = fallback

    def describe(self) -> str:
        if self.fallback is not None:
            return (
                f"{self.primary.describe()} "
                f"(fallback: {self.fallback.describe()})"
            )
        return self.primary.describe()

    def is_available(self) -> bool:
        try:
            if self.primary.is_available():
                return True
        except Exception:
            pass
        if self.fallback is not None:
            try:
                return bool(self.fallback.is_available())
            except Exception:
                return False
        return False

    def ask(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        response = self._try_primary(
            self.primary.ask, user_input, memory, context=context
        )
        if response is None and self.fallback is not None:
            logger.warning("Primary provider failed — using Groq fallback.")
            return self.fallback.ask(user_input, memory, context=context)
        return response

    def ask_stream(
        self,
        user_input: str,
        memory=None,
        on_sentence: Optional[Callable[[str], None]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        response = self._try_primary(
            self.primary.ask_stream,
            user_input,
            memory,
            on_sentence=on_sentence,
            context=context,
        )
        if response is None and self.fallback is not None:
            logger.warning("Primary provider failed — using Groq fallback.")
            return self.fallback.ask_stream(
                user_input, memory, on_sentence=on_sentence, context=context
            )
        return response

    def _try_primary(self, fn, user_input, memory, **kwargs):
        """Run *fn* on the primary provider, catching any exception so a
        broken primary never crashes the assistant."""
        try:
            return fn(user_input, memory, **kwargs)
        except Exception as e:
            logger.warning(f"Primary provider error: {e}")
            return None

    def warmup(self) -> None:
        """Pre-load the primary (local) model in the background."""
        try:
            self.primary.warmup()
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")


def _groq_fallback() -> "LLMProvider | None":
    """Build a GroqClient only when an API key is configured."""
    try:
        from config import groq_config

        if not (groq_config.API_KEY or "").strip():
            return None
    except Exception:
        return None
    try:
        from brain.groq_client import GroqClient

        return GroqClient()
    except Exception as e:
        logger.warning(f"Groq fallback unavailable: {e}")
        return None


def create_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Factory: build the configured AI provider.

    ``name`` defaults to jarvis_config.AI_PROVIDER (env AI_PROVIDER).

    * ``ollama`` (default) — OllamaClient; when GROQ_API_KEY is set it
      is wrapped in a FallbackProvider so Groq steps in if Ollama fails.
    * ``groq`` — GroqClient alone (no key = a provider that is never
      available; callers degrade gracefully).

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

        primary = OllamaClient()
        fallback = _groq_fallback()
        if fallback is not None:
            logger.info(
                "Ollama primary with Groq fallback enabled."
            )
            return FallbackProvider(primary, fallback)
        return primary

    if name == "groq":
        from brain.groq_client import GroqClient

        return GroqClient()

    raise ValueError(
        f"Unknown AI provider '{name}'. Supported: ollama, groq"
    )
