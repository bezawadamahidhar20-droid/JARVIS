"""Local Ollama chat client for Qwen3.

Talks to Ollama's ``/api/chat`` endpoint with the FULL conversation history
so the model keeps context across turns. Every failure mode — server down,
timeout, model missing, non-200 response, malformed JSON, empty reply — is
caught and surfaced as a typed exception with a friendly message, so the
main loop can reply gracefully instead of crashing.
"""

import re

import requests

import config
from utils import logger

# ── Typed exceptions ──────────────────────────────────────────────────────────

class OllamaUnavailableError(Exception):
    """Ollama server is not reachable at all (down / timeout / no network)."""


class ModelMissingError(Exception):
    """Ollama is up, but the configured model is not installed."""


class OllamaResponseError(Exception):
    """Ollama returned a non-success, malformed, or empty response."""


# ── Thinking-token cleanup ────────────────────────────────────────────────────

# Qwen3 emits its "reasoning" in a dedicated `thinking` field, but some
# builds/settings also wrap it inline in special tokens. We strip both forms
# so the user only ever hears the final answer, never the internal monologue.
_THINK_RE = re.compile(
    r"<\|?(?:think|/think)\|?>.*?<\|?/think\|?>"
    r"|<\|?think\|?>"
    r"|<\|?/think\|?>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _clean_thinking(content: str) -> str:
    """Remove Qwen3 thinking sections and stray markers from *content*."""
    if not content:
        return ""
    return _THINK_RE.sub("", content).strip()


class OllamaClient:
    """Thin wrapper around ``POST {base_url}/api/chat``."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT,
        temperature: float = config.OLLAMA_TEMPERATURE,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: int = timeout
        self.temperature: float = temperature
        self.system_prompt: str = system_prompt

    # ── Availability probe ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Quick ping to check whether Ollama is up (never raises)."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Generation ────────────────────────────────────────────────────────────

    def ask(self, user_input: str, memory) -> str:
        """Send *user_input* (plus full memory context) to Qwen3.

        Parameters
        ----------
        user_input:
            The user's current utterance.
        memory:
            A :class:`brain.memory.ConversationMemory` instance. Its
            ``get_context_for_ollama`` returns the system prompt + history.

        Returns
        -------
        The cleaned assistant reply (thinking tokens stripped).

        Raises
        ------
        OllamaUnavailableError, ModelMissingError, OllamaResponseError
        """
        messages = memory.get_context_for_ollama(self.system_prompt)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
            # Keep the model resident between turns so follow-up questions
            # don't trigger a slow (30s+) reload on every exchange.
            "keep_alive": "30m",
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                "Start it (e.g. 'ollama serve' or the Ollama app) and try again."
            )
        except requests.exceptions.Timeout:
            raise OllamaUnavailableError(
                f"Ollama did not answer within {self.timeout}s. "
                "It may still be loading the model — try again."
            )
        except requests.exceptions.RequestException as exc:
            raise OllamaResponseError(f"Ollama request failed: {exc}")

        # Ollama reports a missing model with HTTP 404 / a "model not found"
        # body. Surface a fixable message instead of a generic HTTP error.
        if resp.status_code == 404 or "model not found" in resp.text.lower():
            raise ModelMissingError(
                f"Model '{self.model}' is not installed on this Ollama. "
                f"Install it once with:  ollama pull {self.model}"
            )
        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError:
            raise OllamaResponseError("Ollama returned a non-JSON response.")

        message = data.get("message") or {}
        content = _clean_thinking(message.get("content") or "")

        # If content was empty (some Qwen3 responses only carry `thinking`),
        # fall back to the thinking field — a bare reasoning-only reply is
        # still better than nothing.
        if not content:
            content = _clean_thinking(message.get("thinking") or "")

        if not content:
            raise OllamaResponseError("Ollama returned an empty response.")

        logger.info(f"[AI] replied ({len(content)} chars)")
        return content