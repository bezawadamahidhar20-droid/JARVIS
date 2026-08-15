"""Local Ollama client for Qwen3 via the /api/generate endpoint.

Multi-turn conversation memory
-------------------------------
``OllamaClient`` maintains a sliding window of the last *N* user/assistant
turns (configured via ``OLLAMA_MEMORY_TURNS`` in ``config.py``, default 5).
On every ``generate()`` call the window is prepended to the current prompt so
the model has conversational context — e.g. referents like "it" or "that"
resolve to what was actually discussed, not to a blank slate.

The window is bounded: older turns are automatically evicted, keeping the
prompt size predictable regardless of session length.  Use ``clear_history()``
to reset the window (e.g. after a topic change).
"""

from collections import deque

import requests

import config
from utils import logger

# ── Custom exceptions ─────────────────────────────────────────────────────────

class OllamaUnavailableError(Exception):
    """Ollama server is not reachable at all."""


class ModelMissingError(Exception):
    """Ollama is up but the requested model is not installed."""


class OllamaResponseError(Exception):
    """Ollama returned a non-success response."""


# ── Client ────────────────────────────────────────────────────────────────────

class OllamaClient:
    """Thin wrapper around ``http://localhost:11434/api/generate``.

    Parameters
    ----------
    url:
        Full URL of the Ollama generate endpoint.
    model:
        Name of the model to run (e.g. ``"qwen3:8b"``).
    timeout:
        HTTP request timeout in seconds.
    system_prompt:
        Injected as the ``system`` field in every request.
    num_predict:
        Maximum tokens to generate; keeps replies voice-friendly.
    keep_alive:
        How long Ollama should keep the model loaded between calls.
    memory_turns:
        Number of recent ``(user, assistant)`` pairs to prepend as context.
        Set to ``0`` to disable multi-turn memory entirely.
    """

    def __init__(
        self,
        url: str = config.OLLAMA_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT,
        system_prompt: str = config.SYSTEM_PROMPT,
        num_predict: int = config.OLLAMA_NUM_PREDICT,
        keep_alive: str = config.OLLAMA_KEEP_ALIVE,
        memory_turns: int = config.OLLAMA_MEMORY_TURNS,
    ) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.memory_turns = memory_turns

        # Sliding window of recent (user, assistant) turns.
        # ``deque(maxlen=N)`` automatically evicts the oldest entry when full,
        # so the window never grows beyond ``memory_turns`` pairs.
        self.history: deque[tuple[str, str]] = deque(maxlen=memory_turns)

    # ── Availability check ────────────────────────────────────────────────────

    def _tags_url(self) -> str:
        return self.url.rsplit("/api/", 1)[0] + "/api/tags"

    def check_available(self) -> None:
        """Raise a clear error if Ollama is down or the model is missing."""
        try:
            resp = requests.get(self._tags_url(), timeout=5)
        except requests.exceptions.RequestException:
            raise OllamaUnavailableError(
                "Ollama is not running. Start it with 'ollama serve' "
                "(or the Ollama app) and try again."
            )

        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama tags endpoint returned HTTP {resp.status_code}."
            )

        try:
            models = [m["name"] for m in resp.json().get("models", [])]
        except (ValueError, KeyError):
            raise OllamaResponseError("Could not parse model list from Ollama.")

        if not any(self.model in name for name in models):
            raise ModelMissingError(
                f"Model '{self.model}' is not installed. "
                f"Run:  ollama pull {self.model}"
            )

        logger.ok(f"Ollama ready (model '{self.model}')")

    # ── History management ────────────────────────────────────────────────────

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Append one completed exchange to the sliding-window history.

        Both sides are stripped of whitespace.  Empty strings are not stored
        (a blank assistant reply would pollute the context without adding value).
        """
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if user_text and assistant_text:
            self.history.append((user_text, assistant_text))

    def clear_history(self) -> None:
        """Discard all remembered turns, resetting the context window."""
        self.history.clear()

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(self, prompt: str) -> str:
        """Return *prompt* prefixed with the sliding-window conversation history.

        Format::

            User: <turn 1 user>
            JARVIS: <turn 1 assistant>
            User: <turn 2 user>
            JARVIS: <turn 2 assistant>
            ...
            User: <current prompt>

        When the history is empty the raw prompt is returned unchanged so there
        is no overhead for single-turn queries.
        """
        if not self.history:
            return prompt

        lines: list[str] = []
        for user_msg, assistant_msg in self.history:
            lines.append(f"User: {user_msg}")
            lines.append(f"JARVIS: {assistant_msg}")
        lines.append(f"User: {prompt}")
        return "\n".join(lines)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Qwen3 and return the model's text response.

        The prompt is automatically prefixed with up to ``memory_turns`` prior
        user/assistant pairs so the model has conversational context — e.g.
        follow-up questions like "what else can you tell me about it?" will
        correctly resolve "it" to the topic discussed in the previous turn.

        Raises
        ------
        OllamaUnavailableError
            If Ollama is not running or times out.
        ModelMissingError
            If the configured model is not installed in Ollama.
        OllamaResponseError
            For any other non-200 or malformed response.
        """
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(prompt),
            "system": self.system_prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": False,
            "options": {
                "num_predict": self.num_predict,
            },
        }

        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise OllamaUnavailableError(
                "Cannot connect to Ollama at http://localhost:11434. Is it running?"
            )
        except requests.exceptions.Timeout:
            raise OllamaUnavailableError(
                f"Ollama did not respond within {self.timeout}s."
            )
        except requests.exceptions.RequestException as exc:
            raise OllamaResponseError(f"Ollama request failed: {exc}")

        if resp.status_code == 404 or "model not found" in resp.text.lower():
            raise ModelMissingError(
                f"Model '{self.model}' not found on this Ollama. "
                f"Run:  ollama pull {self.model}"
            )
        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError:
            raise OllamaResponseError("Ollama returned non-JSON output.")

        text: str = data.get("response", "").strip()
        if not text:
            raise OllamaResponseError("Ollama returned an empty response.")
        return text
