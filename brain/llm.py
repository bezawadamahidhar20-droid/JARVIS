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

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

from config import groq_config, jarvis_config
from utils.logger import get_logger

logger = get_logger("llm")

__all__ = [
    "LLMProvider",
    "FallbackProvider",
    "stream_sentences_async",
    "create_provider",
]


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

    async def ask_stream_async(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ):
        """Optional async streaming: yield each finished sentence as it
        is generated, so TTS can start (and barge-in can cancel) while
        the rest of the answer is still being written.

        Providers without a native async path are bridged automatically
        by the orchestrator (see ``stream_sentences_async``). Not an
        abstract method on purpose — sync-only fakes stay compatible.
        """
        raise NotImplementedError(
            f"{self.name} does not implement ask_stream_async"
        )

    def warmup(self) -> None:
        """Optional: pre-load the model so the first real request is fast."""

    def describe(self) -> str:
        """Short status string for `jarvis --doctor`."""
        return self.name


async def stream_sentences_async(
    provider: LLMProvider,
    user_input: str,
    memory=None,
    context: Optional[str] = None,
):
    """Yield response sentences from *any* provider, async.

    Uses ``ask_stream_async`` when the provider has a native async
    path; otherwise bridges the sync ``ask_stream`` over a worker
    thread + asyncio.Queue so the event loop stays responsive either
    way.

    Yields:
        str: each finished sentence
    """
    native = getattr(provider, "ask_stream_async", None)
    if native is not None and not getattr(
        native, "_is_sync_bridge", False
    ):
        try:
            kwargs: dict = {}
            if context is not None:
                kwargs["context"] = context
            async for sentence in native(user_input, memory, **kwargs):
                yield sentence
            return
        except NotImplementedError:
            pass  # fall through to the bridge
        except Exception as e:
            logger.warning(
                f"Async streaming failed ({e}); bridging sync stream."
            )

    queue: "asyncio.Queue" = asyncio.Queue()
    done = threading.Event()
    result_holder: list = []

    def _produce() -> None:
        def _on_sentence(s: str) -> None:
            try:
                queue.put_nowait(("sentence", s))
            except Exception:
                pass

        try:
            kwargs = {}
            if context is not None:
                kwargs["context"] = context
            result = provider.ask_stream(
                user_input, memory, on_sentence=_on_sentence, **kwargs
            )
            result_holder.append(result)
        except Exception as e:
            logger.error(f"Streaming provider error: {e}")
        finally:
            done.set()
            try:
                queue.put_nowait(("done", None))
            except Exception:
                pass

    threading.Thread(
        target=_produce, name="jarvis-sync-stream", daemon=True
    ).start()

    while True:
        kind, value = await queue.get()
        if kind == "done":
            break
        yield value
    done.wait(timeout=2.0)


# Mark the bridge helper so a provider that *delegates* to it (see
# FallbackProvider below) is never mistaken for a native async path.
stream_sentences_async._is_sync_bridge = True  # type: ignore[attr-defined]


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

    async def ask_stream_async(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ):
        """Async streaming with primary -> fallback ordering.

        Tries the primary's native async path first; if that fails or
        produces nothing, the fallback's sync stream is bridged.
        """
        primary_async = getattr(self.primary, "ask_stream_async", None)
        if primary_async is not None:
            produced = False
            try:
                async for sentence in primary_async(
                    user_input, memory, context=context
                ):
                    produced = True
                    yield sentence
            except Exception as e:
                logger.warning(
                    f"Primary async stream failed ({e}); using fallback."
                )
            if produced:
                return
        if self.fallback is not None:
            logger.warning("Primary provider failed — using Groq fallback.")
            async for sentence in stream_sentences_async(
                self.fallback, user_input, memory, context=context
            ):
                yield sentence

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

    def switch_model(self, model: str) -> str:
        """Switch the primary (local) model at runtime."""
        fn = getattr(self.primary, "switch_model", None)
        if fn is None:
            raise NotImplementedError(
                "primary provider cannot switch models at runtime"
            )
        return fn(model)


def _groq_fallback() -> "LLMProvider | None":
    """Build a GroqClient only when an API key is configured."""
    if not (groq_config.API_KEY or "").strip():
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
        name = jarvis_config.AI_PROVIDER

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
