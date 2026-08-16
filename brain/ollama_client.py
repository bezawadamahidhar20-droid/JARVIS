"""
brain/ollama_client.py — Ollama integration (configurable model)

Implements the LLMProvider interface (brain/llm.py) so JARVIS can
swap in other providers later without rewriting the assistant.

Speed optimizations (qwen3:8b / llama3.2 on CPU):
  1. keep_alive: 30m       — keeps model loaded in RAM
  2. num_predict: 150      — limits response length
  3. num_ctx: 2048         — smaller context = faster processing
  4. num_gpu: 99           — offload to GPU if available
  5. Short system prompt   — fewer tokens processed per call
  6. stream: True          — tokens arrive as they are generated, so
                             the first sentence can be spoken while the
                             rest of the answer is still being written
  7. think: False          — Qwen3's reasoning tokens are suppressed,
                             cutting response latency dramatically

ask_stream() emits each complete sentence to on_sentence() as it
finishes; ask() remains for --text mode / non-streaming callers.
"""

import re
import json
import time
import requests
from typing import Optional, List, Dict, Callable
from utils.logger import get_logger
from brain.llm import LLMProvider

logger = get_logger("ollama_client")

# ── Load config safely ────────────────────────────────────────
try:
    from config import ollama_config, jarvis_config
    OLLAMA_BASE_URL    = ollama_config.BASE_URL
    OLLAMA_MODEL       = ollama_config.MODEL
    OLLAMA_TIMEOUT     = ollama_config.TIMEOUT
    OLLAMA_TEMP        = ollama_config.TEMPERATURE
    OLLAMA_STREAM      = ollama_config.STREAM
    OLLAMA_NUM_PREDICT = ollama_config.NUM_PREDICT
    OLLAMA_NUM_CTX     = ollama_config.NUM_CTX
    OLLAMA_KEEP_ALIVE  = ollama_config.KEEP_ALIVE
    OLLAMA_NUM_GPU     = ollama_config.NUM_GPU
    OLLAMA_THINK       = ollama_config.THINK
    OWNER              = jarvis_config.OWNER
except Exception as e:
    logger.warning(f"Config load failed, using defaults: {e}")
    OLLAMA_BASE_URL    = "http://localhost:11434"
    OLLAMA_MODEL       = "qwen3:8b"
    OLLAMA_TIMEOUT     = 120
    OLLAMA_TEMP        = 0.7
    OLLAMA_STREAM      = True
    OLLAMA_NUM_PREDICT = 150
    OLLAMA_NUM_CTX     = 2048
    OLLAMA_KEEP_ALIVE  = "30m"
    OLLAMA_NUM_GPU     = 99
    OLLAMA_THINK       = False
    OWNER              = "Sir"

# ── Short system prompt (<60 words, fewer tokens = faster) ────
SYSTEM_PROMPT = (
    f"You are JARVIS, a concise British AI butler. "
    f"Address the user as {OWNER}. "
    f"Answer in 1 to 3 short sentences. "
    f"Be direct and natural. "
    f"No bullet points. No preamble. "
    f"Never refuse to answer a normal question."
)

# System prompt used when answering from verified web-search results.
# ``{context}`` is replaced with the formatted search results.
SEARCH_SYSTEM_PROMPT = (
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

# A sentence ends at . ! ? (followed by whitespace or end) or a newline.
# NOTE: must be "\n" (real newline) — "\\n" would match a literal
# backslash-n and newline-splitting would silently never fire.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|\n")
# Safety valve: if a chunk of text is this long with no sentence end,
# emit it anyway so speech never stalls on run-on text.
MAX_PARTIAL_CHARS = 200


class OllamaClient(LLMProvider):
    """
    Client for a local Ollama server (configurable model).
    Optimized for low latency voice assistant use.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: int = None,
        temperature: float = None,
        stream: bool = None,
        num_predict: int = None,
        num_ctx: int = None,
        keep_alive=None,
        num_gpu: int = None,
        think: bool = None
    ):
        self.base_url    = base_url      or OLLAMA_BASE_URL
        self.model       = model         or OLLAMA_MODEL
        self.timeout     = timeout       or OLLAMA_TIMEOUT
        self.temperature = temperature   if temperature is not None else OLLAMA_TEMP
        self.stream      = stream        if stream is not None else OLLAMA_STREAM
        self.num_predict = num_predict   if num_predict is not None else OLLAMA_NUM_PREDICT
        self.num_ctx     = num_ctx       if num_ctx is not None else OLLAMA_NUM_CTX
        self.keep_alive  = keep_alive    or OLLAMA_KEEP_ALIVE
        self.num_gpu     = num_gpu       if num_gpu is not None else OLLAMA_NUM_GPU
        self.think       = think         if think is not None else OLLAMA_THINK
        self._verify_connection()

    def describe(self) -> str:
        """Short status string for `jarvis --doctor`."""
        return f"{self.model} @ {self.base_url}"

    # ── Connection check ──────────────────────────────────────

    def _verify_connection(self) -> None:
        """Check Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                data   = response.json()
                models = data.get("models", [])
                names  = [m.get("name", "") for m in models]
                logger.info(
                    f"Ollama connected. "
                    f"Available models: {names}"
                )
                base = self.model.split(":")[0]
                if not any(base in n for n in names):
                    logger.warning(
                        f"Model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    )
            else:
                logger.warning(
                    f"Ollama status: {response.status_code}"
                )
        except requests.ConnectionError:
            logger.error(
                f"Cannot reach Ollama at {self.base_url}. "
                "Run: ollama serve"
            )
        except Exception as e:
            logger.error(f"Connection check error: {e}")

    def is_available(self) -> bool:
        """Quick ping to check if Ollama is reachable."""
        try:
            r = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    # ── Warm-up ───────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Pre-load the model into RAM by sending a tiny request
        (num_predict=1). Call this in a background thread on
        startup so the first real question has no cold-start delay.
        """
        logger.info(
            f"Warming up {self.model} (loading into RAM)..."
        )
        print(f"[Warming up {self.model}...]")
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are JARVIS."
                    },
                    {
                        "role": "user",
                        "content": "hi"
                    }
                ],
                "stream": False,
                "keep_alive": self.keep_alive,
                # Match the real request: suppress Qwen3 reasoning so the
                # loaded state is identical (think:true would load extra
                # reasoning machinery / template state).
                "think": self.think if self.think is not None else False,
                "options": {
                    "num_predict": 1,
                    # Must match the real request's num_ctx, otherwise
                    # Ollama re-allocates the context on the first real
                    # question and the warm-up benefit is lost.
                    "num_ctx": self.num_ctx,
                    "temperature": 0.1,
                }
            }
            requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            logger.info("Model warm-up complete.")
            print("[Model ready]")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    # ── Streaming ask (voice mode) ────────────────────────────

    def ask_stream(
        self,
        user_input: str,
        memory=None,
        on_sentence: Callable[[str], None] = None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask llama and stream tokens back as they are generated.

        Each time a complete sentence is detected, it is cleaned and
        handed to ``on_sentence`` (e.g. a non-blocking TTS) so speech
        starts while the rest of the answer is still being written.

        Args:
            user_input  : the user's question or statement
            memory      : ConversationMemory instance
            on_sentence : callback(sentence_text) for each complete sentence
            context     : optional verified search results to answer from

        Returns:
            str  : full JARVIS response text
            None : if the request failed
        """
        if not user_input or not user_input.strip():
            return None

        t_build = time.perf_counter()
        messages = self._build_messages(user_input, memory, context=context)
        payload = self._build_payload(messages, stream=True)
        logger.debug(
            f"[timing] prompt build {(time.perf_counter() - t_build) * 1000:.0f}ms "
            f"({len(messages)} messages)"
        )

        logger.info(
            f"Asking [{self.model}]: '{user_input[:60]}'"
        )

        full, _ = self._stream_generate(payload, on_sentence)

        if not full or not full.strip():
            logger.error("Ollama returned empty streamed response.")
            return None

        return self._clean_response(full)

    # ── Non-streaming ask (--text mode / back-compat) ─────────

    def ask(
        self,
        user_input: str,
        memory=None,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a message to llama and return the full response.
        Includes conversation history for context.

        Args:
            user_input : the user's question or statement
            memory     : ConversationMemory instance
            context    : optional verified search results to answer from

        Returns:
            str  : JARVIS response text
            None : if the request failed
        """
        if not user_input or not user_input.strip():
            return None

        t_build = time.perf_counter()
        messages = self._build_messages(user_input, memory, context=context)
        payload = self._build_payload(messages, stream=False)
        logger.debug(
            f"[timing] prompt build {(time.perf_counter() - t_build) * 1000:.0f}ms "
            f"({len(messages)} messages)"
        )

        start_time = time.perf_counter()

        try:
            logger.info(
                f"Asking [{self.model}]: '{user_input[:60]}'"
            )

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            elapsed = time.perf_counter() - start_time
            data    = response.json()

            content = (
                data
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                logger.error("Ollama returned empty response.")
                return None

            # Clean markdown artifacts that sound bad when spoken
            content = self._clean_response(content)

            logger.info(
                f"[timing] response in {elapsed:.1f}s | "
                f"{len(content)} chars"
            )
            print(f"[Response time: {elapsed:.1f}s]")

            return content

        except requests.ConnectionError:
            logger.error("Lost connection to Ollama.")
            return None

        except requests.Timeout:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Ollama timed out after {elapsed:.0f}s. "
                "Try restarting Ollama."
            )
            return None

        except requests.HTTPError as e:
            logger.error(
                f"Ollama HTTP error "
                f"{e.response.status_code}: {e}"
            )
            return None

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse Ollama JSON: {e}"
            )
            return None

        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            return None

    # ── Streaming internals ───────────────────────────────────

    def _stream_generate(
        self,
        payload: dict,
        on_sentence: Callable[[str], None]
    ):
        """
        Stream NDJSON lines from Ollama, split them into sentences and
        hand each one to on_sentence as it completes. Returns
        (full_text, timing_dict).
        """
        start = time.perf_counter()
        first_token_at = None
        full = ""
        buffer = ""
        tokens = 0

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )

            if resp.status_code != 200:
                try:
                    body = resp.json()
                    err = body.get("error", resp.text[:120])
                except Exception:
                    err = resp.text[:120]
                logger.error(
                    f"Ollama stream HTTP {resp.status_code}: {err}"
                )
                return None, None

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    logger.error(f"Ollama stream error: {data['error']}")
                    return None, None

                msg = data.get("message", {}) or {}
                chunk = msg.get("content", "") or ""
                if chunk:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    tokens += 1
                    full += chunk
                    buffer += chunk

                    # Emit complete sentences as soon as they finish.
                    sentences, buffer = self._extract_sentences(buffer)
                    for s in sentences:
                        clean = self._clean_response(s)
                        if clean and on_sentence:
                            on_sentence(clean)

                    # Safety valve for run-on text without punctuation.
                    if len(buffer) >= MAX_PARTIAL_CHARS:
                        clean = self._clean_response(buffer)
                        if clean and on_sentence:
                            on_sentence(clean)
                        buffer = ""

                if data.get("done"):
                    break

            # Flush any trailing partial sentence.
            if buffer.strip():
                clean = self._clean_response(buffer)
                if clean and on_sentence:
                    on_sentence(clean)
                buffer = ""

            resp.close()

        except requests.ConnectionError:
            logger.error("Lost connection to Ollama.")
            return None, None
        except requests.Timeout:
            logger.error(
                f"Ollama stream timed out after {self.timeout}s."
            )
            return None, None
        except requests.RequestException as e:
            logger.error(f"Ollama stream request error: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            return None, None

        elapsed = time.perf_counter() - start
        first = (first_token_at - start) if first_token_at else elapsed
        logger.info(
            f"[timing] first token {first:.1f}s | "
            f"total {elapsed:.1f}s | {tokens} tokens"
        )
        return full, {"first_token": first, "total": elapsed, "tokens": tokens}

    def _extract_sentences(self, buffer: str):
        """
        Pull complete sentences out of the streaming buffer.

        Returns (sentences, remainder). A sentence ends at . ! ?
        (followed by whitespace or end-of-text) or a newline.
        """
        sentences = []
        while True:
            m = _SENTENCE_END_RE.search(buffer)
            if not m:
                break
            end = m.end()
            sentence = buffer[:end].rstrip("\n").strip()
            buffer = buffer[end:]
            if sentence:
                sentences.append(sentence)
        return sentences, buffer

    # ── Payload builders ──────────────────────────────────────

    def _build_messages(
        self,
        user_input: str,
        memory,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the message list for Ollama.

        main() already stores the user message in memory before asking,
        so if it is the last entry we reuse it instead of duplicating
        it (fixes the old duplicate-user-message bug that doubled the
        prompt tokens on every request).

        When ``context`` (verified search results) is provided, the
        system prompt instructs the model to answer only from it.
        """
        user_text = user_input.strip()

        if context:
            system_prompt = SEARCH_SYSTEM_PROMPT.format(context=context)
        else:
            system_prompt = SYSTEM_PROMPT

        if memory is not None:
            ctx = memory.get_context_for_ollama(system_prompt)
            if (
                ctx
                and ctx[-1]["role"] == "user"
                and ctx[-1]["content"].strip() == user_text
            ):
                return ctx
            ctx.append({"role": "user", "content": user_text})
            return ctx

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool
    ) -> dict:
        """Build the optimized /api/chat payload for llama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,

            # Keep model in RAM — prevents cold reload between questions.
            "keep_alive": self.keep_alive,

            "options": {
                "temperature": self.temperature,

                # Limit response length (~3 sentences) — prevents
                # rambling answers that keep the user waiting.
                "num_predict": self.num_predict,

                # Smaller context window = faster prompt processing.
                "num_ctx": self.num_ctx,

                # Offload all layers to GPU if available.
                # Ollama silently ignores this if no GPU is present.
                "num_gpu": self.num_gpu,

                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        # Qwen3 emits reasoning tokens unless "think" is explicitly
        # disabled. Suppress them for fast voice replies.
        if self.think is not None:
            payload["think"] = self.think
        return payload

    # ── Response cleaning ─────────────────────────────────────

    def _clean_response(self, text: str) -> str:
        """
        Strip markdown artifacts that sound bad when spoken aloud.
        """
        # Remove any XML-style tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove markdown bold and italic markers
        text = re.sub(r'\*+', '', text)

        # Remove markdown headers
        text = re.sub(
            r'^#+\s*', '', text,
            flags=re.MULTILINE
        )
        # Remove backtick code markers
        text = re.sub(r'`+', '', text)

        # Collapse extra blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


# ── Module-level convenience functions ────────────────────────

def ask_ollama(
    user_input: str,
    memory=None,
    client: OllamaClient = None
) -> Optional[str]:
    """
    Convenience wrapper. Creates a client if none given.
    """
    if client is None:
        client = OllamaClient()
    return client.ask(user_input, memory)


def ask_ollama_stream(
    user_input: str,
    memory=None,
    on_sentence: Callable[[str], None] = None,
    client: OllamaClient = None
) -> Optional[str]:
    """
    Convenience streaming wrapper. Creates a client if none given.
    """
    if client is None:
        client = OllamaClient()
    return client.ask_stream(user_input, memory, on_sentence)