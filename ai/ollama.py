"""Local Ollama client for Qwen3 via the /api/generate endpoint."""

from collections import deque
from typing import Deque, Tuple

import requests

import config
from utils import logger


class OllamaUnavailableError(Exception):
    """Ollama server is not reachable at all."""


class ModelMissingError(Exception):
    """Ollama is up but the requested model is not installed."""


class OllamaResponseError(Exception):
    """Ollama returned a non-success response."""


class OllamaClient:
    """Thin wrapper around http://localhost:11434/api/generate."""

    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.OLLAMA_MODEL,
                 timeout: int = config.OLLAMA_TIMEOUT,
                 system_prompt: str = config.SYSTEM_PROMPT,
                 num_predict: int = config.OLLAMA_NUM_PREDICT,
                 keep_alive: str = config.OLLAMA_KEEP_ALIVE,
                 memory_turns: int = config.OLLAMA_MEMORY_TURNS):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        # Sliding window of recent (user, assistant) turns. Oldest entries
        # are dropped first so the prompt stays bounded.
        self.memory_turns = memory_turns
        self.history: Deque[Tuple[str, str]] = deque(maxlen=memory_turns)

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
                f"Model '{self.model}' is not installed. Run:  ollama pull {self.model}"
            )
        logger.ok(f"Ollama ready (model '{self.model}')")

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Record one completed user/assistant turn in the sliding window."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if user_text and assistant_text:
            self.history.append((user_text, assistant_text))

    def clear_history(self) -> None:
        """Drop all remembered turns."""
        self.history.clear()

    def _build_prompt(self, prompt: str) -> str:
        """Prefix the current prompt with recent conversation context."""
        if not self.history:
            return prompt
        lines = []
        for user_text, assistant_text in self.history:
            lines.append(f"User: {user_text}")
            lines.append(f"JARVIS: {assistant_text}")
        lines.append(f"User: {prompt}")
        return "\n".join(lines)

    def generate(self, prompt: str) -> str:
        """Send a prompt to Qwen3. Returns the model's text response.

        The prompt is prefixed with up to `memory_turns` prior
        user/assistant pairs, giving the model conversational context.
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
                "Cannot connect to Ollama at http://localhost:11434. "
                "Is it running?"
            )
        except requests.exceptions.Timeout:
            raise OllamaUnavailableError(
                f"Ollama did not respond within {self.timeout}s."
            )
        except requests.exceptions.RequestException as exc:
            raise OllamaResponseError(f"Ollama request failed: {exc}")

        if resp.status_code == 404 or "model not found" in resp.text.lower():
            raise ModelMissingError(
                f"Model '{self.model}' not found on this Ollama. Run:  ollama pull {self.model}"
            )
        if resp.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError:
            raise OllamaResponseError("Ollama returned non-JSON output.")

        text = data.get("response", "").strip()
        if not text:
            raise OllamaResponseError("Ollama returned an empty response.")
        return text