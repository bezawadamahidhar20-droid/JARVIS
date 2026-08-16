"""
brain/groq_client.py — optional Groq fallback LLM provider.

JARVIS's primary brain is the local Ollama server. When Ollama is
unavailable (not running, model missing, network down), a *configured*
Groq account can step in so conversational answers keep working.

Rules:
  * Groq is strictly optional: with no GROQ_API_KEY it reports itself
    unavailable and is never constructed into the pipeline.
  * The API key is only ever read from the environment / .env
    (config.py) — never hardcoded and never logged.
  * ask() / ask_stream() return None on any failure so callers fall
    back gracefully instead of crashing.
  * Streaming uses the same sentence splitter as Ollama
    (brain/text_utils.py) so TTS still starts on the first sentence.
"""

from typing import Callable, List, Optional

import requests

from brain.llm import LLMProvider
from brain.text_utils import clean_response, split_into_sentences
from utils.logger import get_logger

logger = get_logger("groq_client")

# ── Load config safely ────────────────────────────────────────
try:
    from config import groq_config, jarvis_config

    GROQ_API_KEY = groq_config.API_KEY
    GROQ_MODEL = groq_config.MODEL
    GROQ_TIMEOUT = groq_config.TIMEOUT
    GROQ_TEMPERATURE = groq_config.TEMPERATURE
    GROQ_MAX_TOKENS = groq_config.MAX_TOKENS
    OWNER = jarvis_config.OWNER
except Exception as e:
    logger.warning(f"Config load failed, using defaults: {e}")
    GROQ_API_KEY = ""
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_TIMEOUT = 60
    GROQ_TEMPERATURE = 0.7
    GROQ_MAX_TOKENS = 200
    OWNER = "Sir"

_SYSTEM_PROMPT = (
    f"You are JARVIS, a concise British AI butler. "
    f"Address the user as {OWNER}. "
    "Answer in 1 to 3 short sentences. "
    "Be direct and natural. "
    "No bullet points. No preamble. "
    "Never refuse to answer a normal question."
)

_SEARCH_SYSTEM_PROMPT = (
    f"You are JARVIS, a concise British AI butler. "
    f"Address the user as {OWNER}. "
    "Answer the user's question using ONLY the verified search "
    "information provided below. Do not invent facts. Do not use your "
    "training memory to override the search information. If the "
    "information does not contain enough evidence to answer, say the "
    "information could not be verified. If sources disagree, state the "
    "uncertainty. Answer in 1 to 3 short sentences. No bullet points. "
    "No preamble.\n\n"
    "VERIFIED SEARCH INFORMATION:\n{context}"
)

# Safety valve: emit a run-on chunk this long even without a sentence end.
MAX_PARTIAL_CHARS = 200


class GroqClient(LLMProvider):
    """Groq LLM API client (OpenAI-compatible chat completions)."""

    name = "groq"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else GROQ_API_KEY).strip()
        self.model = model or GROQ_MODEL
        self.timeout = timeout if timeout is not None else GROQ_TIMEOUT
        self.temperature = (
            temperature if temperature is not None else GROQ_TEMPERATURE
        )
        self.max_tokens = max_tokens if max_tokens is not None else GROQ_MAX_TOKENS
        if not self.is_configured():
            logger.info(
                "Groq not configured (no GROQ_API_KEY) — skipping fallback."
            )

    def describe(self) -> str:
        if self.is_configured():
            return f"groq ({self.model})"
        return "groq (no API key — disabled)"

    def is_configured(self) -> bool:
        """True when an API key is present. Never exposes the key."""
        return bool(self.api_key)

    def is_available(self) -> bool:
        """Groq counts as available when configured (a live ping would
        burn API credits on every check)."""
        return self.is_configured()

    # ── Request plumbing ──────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(
        self,
        user_input: str,
        memory,
        context: Optional[str] = None,
    ) -> List[dict]:
        """Build the chat message list (same shape as the Ollama client)."""
        user_text = user_input.strip()
        system = (
            _SEARCH_SYSTEM_PROMPT.format(context=context)
            if context
            else _SYSTEM_PROMPT
        )
        if memory is not None:
            ctx = memory.get_context_for_ollama(system)
            if (
                ctx
                and ctx[-1]["role"] == "user"
                and ctx[-1]["content"].strip() == user_text
            ):
                return ctx
            ctx.append({"role": "user", "content": user_text})
            return ctx
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    def _payload(self, messages: List[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    # ── Non-streaming ─────────────────────────────────────────

    def ask(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        if not user_input or not user_input.strip():
            return None
        if not self.is_configured():
            logger.warning("Groq unavailable — no GROQ_API_KEY set.")
            return None
        try:
            messages = self._build_messages(user_input, memory, context=context)
            resp = requests.post(
                self.ENDPOINT,
                headers=self._headers(),
                json=self._payload(messages, stream=False),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                logger.error("Groq returned an empty response.")
                return None
            return clean_response(content)
        except requests.ConnectionError:
            logger.error("Cannot reach Groq API.")
            return None
        except requests.Timeout:
            logger.error(f"Groq API timed out after {self.timeout}s.")
            return None
        except requests.HTTPError as e:
            logger.error(f"Groq HTTP error {e.response.status_code}.")
            return None
        except Exception as e:
            logger.error(f"Groq request failed: {e}")
            return None

    # ── Streaming ─────────────────────────────────────────────

    def ask_stream(
        self,
        user_input: str,
        memory=None,
        on_sentence: Optional[Callable[[str], None]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        if not user_input or not user_input.strip():
            return None
        if not self.is_configured():
            logger.warning("Groq unavailable — no GROQ_API_KEY set.")
            return None
        try:
            messages = self._build_messages(user_input, memory, context=context)
            resp = requests.post(
                self.ENDPOINT,
                headers=self._headers(),
                json=self._payload(messages, stream=True),
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()

            full = ""
            buffer = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    import json

                    data = json.loads(payload)
                except Exception:
                    continue
                delta = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if not delta:
                    continue
                full += delta
                buffer += delta

                sentences, buffer = split_into_sentences(buffer)
                for s in sentences:
                    clean = clean_response(s)
                    if clean and on_sentence:
                        on_sentence(clean)

                if len(buffer) >= MAX_PARTIAL_CHARS:
                    clean = clean_response(buffer)
                    if clean and on_sentence:
                        on_sentence(clean)
                    buffer = ""

            if buffer.strip():
                clean = clean_response(buffer)
                if clean and on_sentence:
                    on_sentence(clean)

            resp.close()
        except requests.ConnectionError:
            logger.error("Cannot reach Groq API.")
            return None
        except requests.Timeout:
            logger.error(f"Groq API timed out after {self.timeout}s.")
            return None
        except requests.HTTPError as e:
            logger.error(f"Groq HTTP error {e.response.status_code}.")
            return None
        except Exception as e:
            logger.error(f"Groq stream failed: {e}")
            return None

        if not full or not full.strip():
            logger.error("Groq returned an empty streamed response.")
            return None
        return clean_response(full)
