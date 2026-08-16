"""
brain/groq_client.py — Optional Groq fallback LLM provider.
 
[FIX m10] Added ask_stream_async() method for async streaming support.
[FIX m5] Added __all__ exports.
[FIX m1] Removed try/except config fallbacks.
"""
 
import asyncio
from typing import Callable, List, Optional
 
import requests
 
from brain.llm import LLMProvider
from brain.text_utils import clean_response, split_into_sentences
from config import groq_config, jarvis_config
from utils.logger import get_logger
 
__all__ = [
    "GroqClient",
]
 
logger = get_logger("groq_client")
 
# Config values - direct imports
GROQ_API_KEY = groq_config.API_KEY
GROQ_MODEL = groq_config.MODEL
GROQ_TIMEOUT = groq_config.TIMEOUT
GROQ_TEMPERATURE = groq_config.TEMPERATURE
GROQ_MAX_TOKENS = groq_config.MAX_TOKENS
OWNER = jarvis_config.OWNER
 
_SYSTEM_PROMPT = (
    f"You are JARVIS, a concise British AI butler. "
    f"Address the user as {OWNER}. "
    "Answer in 1 to 3 short sentences. "
    "Be direct and natural. No bullet points. No preamble. "
    "Never refuse to answer a normal question."
)
 
_SEARCH_SYSTEM_PROMPT = (
    f"You are JARVIS, a concise British AI butler. "
    f"Address the user as {OWNER}. "
    "Answer the user's question using ONLY the verified search "
    "information provided below. Do not invent facts. "
    "Answer in 1 to 3 short sentences. No bullet points.\n\n"
    "VERIFIED SEARCH INFORMATION:\n{context}"
)
 
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
        self.temperature = temperature if temperature is not None else GROQ_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else GROQ_MAX_TOKENS
        
        if not self.is_configured():
            logger.info("Groq not configured (no GROQ_API_KEY) — skipping fallback.")
 
    def describe(self) -> str:
        if self.is_configured():
            return f"groq ({self.model})"
        return "groq (no API key — disabled)"
 
    def is_configured(self) -> bool:
        return bool(self.api_key)
 
    def is_available(self) -> bool:
        return self.is_configured()
 
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
        user_text = user_input.strip()
        system = (
            _SEARCH_SYSTEM_PROMPT.format(context=context)
            if context
            else _SYSTEM_PROMPT
        )
        if memory is not None:
            ctx = memory.get_context_for_ollama(system)
            if ctx and ctx[-1]["role"] == "user" and ctx[-1]["content"].strip() == user_text:
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
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    import json
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    chunk = delta.get("content", "")
                    if chunk:
                        full += chunk
                        buffer += chunk
                        
                        sentences, remainder = split_into_sentences(buffer)
                        if len(sentences) > 1:
                            for sent in sentences[:-1]:
                                clean = clean_response(sent)
                                if clean and on_sentence:
                                    on_sentence(clean)
                            buffer = remainder or sentences[-1]
                        elif len(buffer) > MAX_PARTIAL_CHARS:
                            if on_sentence:
                                on_sentence(buffer)
                            buffer = ""
                except Exception:
                    continue
 
            if buffer.strip() and on_sentence:
                on_sentence(buffer)
 
            return clean_response(full) if full else None
 
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            return None
 
    # [FIX m10] Added async streaming support
    async def ask_stream_async(
        self,
        user_input: str,
        memory=None,
        on_sentence: Optional[Callable[[str], None]] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Async streaming - uses run_in_executor to wrap sync request.
        This allows FallbackProvider to use async path with Groq.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ask_stream(user_input, memory, on_sentence, context)
        )
 