"""
config.py — Central configuration for JARVIS
Loads all settings from .env file using python-dotenv.

All values use typed safe getters with fallbacks so a malformed or
missing .env value never crashes JARVIS.

Every setting is read from the environment / .env — nothing is
hardcoded in the engine modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (path is absolute, so it works
# regardless of the current working directory).
# override=True: the .env file is authoritative — a stale OLLAMA_MODEL
# exported in a previous terminal session must not silently win.
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# ── Typed getters with safe fallbacks ─────────────────────────
def _env_str(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value.strip() if value else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


# Placeholder values users paste from tutorials / READMEs. Treating them
# as real secrets would make the app *appear* configured while every
# call fails with an auth error — the exact silent failure this file
# exists to prevent. See _env_secret().
_PLACEHOLDER_SECRETS = {
    "your-key-here", "your_key_here", "your key here",
    "changeme", "replace-me", "replace_me",
    "your_api_key_here", "your-api-key-here", "xxx", "sk-xxx",
    "add-your-key", "api-key", "<api-key>", "YOUR_API_KEY",
    "placeholder", "secret", "key", "12345",
}


def _env_secret(key: str) -> str:
    """
    Read an API key from the environment, normalizing placeholder
    values to "" (unconfigured) so the app degrades gracefully instead
    of failing every call with an auth error.

    The original raw value is still visible to validate_config() (which
    reads os.getenv directly) so the user gets a clear warning.
    """
    value = os.getenv(key, "").strip()
    if not value:
        return ""
    if value.lower() in _PLACEHOLDER_SECRETS:
        return ""
    return value


class OllamaConfig:
    # Base URL of the local Ollama server.
    BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    # Model used as the AI brain (configurable; qwen3:8b is the default).
    MODEL: str = _env_str("OLLAMA_MODEL", "qwen3:8b")
    # Circuit breaker: after this many consecutive failures JARVIS
    # fast-fails AI requests ("AI offline") instead of blocking for
    # OLLAMA_TIMEOUT on every request, and auto-recovers once Ollama
    # comes back. 0 disables the breaker.
    CIRCUIT_BREAKER: bool = _env_bool("OLLAMA_CIRCUIT_BREAKER", True)
    # Consecutive failures that open the breaker (fast-fail mode).
    CIRCUIT_THRESHOLD: int = _env_int("OLLAMA_CIRCUIT_THRESHOLD", 3)
    # Seconds the breaker stays open before probing Ollama again.
    CIRCUIT_RECOVERY: float = _env_float("OLLAMA_CIRCUIT_RECOVERY", 30.0)
    # Optional per-mode models, selected by JARVIS_MODEL_MODE:
    #   quality -> QUALITY_MODEL (falling back to MODEL)
    #   fast    -> FAST_MODEL     (falling back to MODEL)
    # Empty values keep the current OLLAMA_MODEL behavior.
    FAST_MODEL: str = _env_str("OLLAMA_FAST_MODEL", "")
    QUALITY_MODEL: str = _env_str("OLLAMA_QUALITY_MODEL", "")
    TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)
    TEMPERATURE: float = _env_float("OLLAMA_TEMPERATURE", 0.7)
    STREAM: bool = _env_bool("OLLAMA_STREAM", True)
    # num_predict accepts the short alias MAX_RESPONSE_TOKENS too.
    # 120 tokens ~= 3-4 short spoken sentences; caps the worst-case
    # wait on slow CPUs without trimming accurate answers.
    NUM_PREDICT: int = _env_int(
        "OLLAMA_NUM_PREDICT",
        _env_int("MAX_RESPONSE_TOKENS", 120),
    )
    NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 2048)
    KEEP_ALIVE: str = _env_str("OLLAMA_KEEP_ALIVE", "30m")
    NUM_GPU: int = _env_int("OLLAMA_NUM_GPU", 99)
    # Disable Qwen3 "thinking" tokens for low latency voice replies.
    THINK: bool = _env_bool("OLLAMA_THINK", False)

    def resolve_model(self, mode: str | None = None) -> str:
        """
        Pick the Ollama model for the current JARVIS_MODEL_MODE.

        * mode == "fast"    -> OLLAMA_FAST_MODEL   (or OLLAMA_MODEL)
        * mode == "quality" -> OLLAMA_QUALITY_MODEL (or OLLAMA_MODEL)
        * anything else     -> OLLAMA_MODEL (the default / current behavior)

        An explicit ``mode`` argument overrides JARVIS_MODEL_MODE (used
        by tests and by callers that need a specific mode).
        """
        if mode is None:
            mode = (jarvis_config.MODEL_MODE or "quality").strip().lower()
        if mode == "fast" and (self.FAST_MODEL or "").strip():
            return self.FAST_MODEL.strip()
        if mode == "quality" and (self.QUALITY_MODEL or "").strip():
            return self.QUALITY_MODEL.strip()
        return (self.MODEL or "qwen3:8b").strip()


class WhisperConfig:
    # Which faster-whisper model to use ("tiny", "base", "small", ...).
    # Smaller = faster; "base" is a good accuracy/speed balance.
    MODEL: str = _env_str("WHISPER_MODEL", "base")
    # "int8" for CPU, "float16" for CUDA.
    COMPUTE_TYPE: str = _env_str("WHISPER_COMPUTE_TYPE", "int8")
    # "cpu", "cuda", or "auto".
    DEVICE: str = _env_str("WHISPER_DEVICE", "cpu")
    # Whisper language code ("en", "es", ...).
    LANGUAGE: str = _env_str("WHISPER_LANGUAGE", "en")
    # 1 = greedy, fast; higher = slower but more accurate.
    BEAM_SIZE: int = _env_int("WHISPER_BEAM_SIZE", 1)
    # Keep the model loaded between utterances (always true for JARVIS).
    PRELOAD: bool = _env_bool("WHISPER_PRELOAD", True)


class STTConfig:
    # Capture settings used by the VAD-based recorder.
    SAMPLE_RATE: int = _env_int("SAMPLE_RATE", 16000)
    # Optional numeric device index; None = system default input.
    INPUT_DEVICE: int | None = (
        _env_int("INPUT_DEVICE", -1) if _env_int("INPUT_DEVICE", -1) >= 0 else None
    )
    LANGUAGE: str = _env_str("STT_LANGUAGE", "en-US")  # informational
    # Seconds to wait for speech to start before giving up.
    TIMEOUT: int = _env_int("STT_TIMEOUT", 5)
    # Max phrase length in seconds before cutting off a long utterance.
    PHRASE_LIMIT: int = _env_int("STT_PHRASE_LIMIT", 10)


class VADConfig:
    # Seconds of ambient noise sampled at startup to calibrate the
    # speech threshold (adaptive VAD). 0 disables calibration.
    CALIBRATE_SECONDS: float = _env_float("VAD_CALIBRATE_SECONDS", 0.6)
    # Speech threshold = max(MIN_THRESHOLD, ambient_rms * MULTIPLIER).
    THRESHOLD_MULTIPLIER: float = _env_float("VAD_THRESHOLD_MULTIPLIER", 3.0)
    # Absolute floor for the speech threshold (RMS).
    MIN_THRESHOLD: float = _env_float("VAD_MIN_THRESHOLD", 120.0)
    # Fallback fixed threshold used when calibration is disabled/fails.
    FIXED_THRESHOLD: float = _env_float("VAD_FIXED_THRESHOLD", 500.0)
    # Seconds of silence that ends a phrase.
    SILENCE_DURATION: float = _env_float("VAD_SILENCE_DURATION", 0.7)
    # Audio chunk length in seconds (0.05 = snappy onset detection).
    CHUNK_DURATION: float = _env_float("VAD_CHUNK_DURATION", 0.05)
    # Ignore utterances shorter than this many speech chunks.
    MIN_SPEECH_CHUNKS: int = _env_int("VAD_MIN_SPEECH_CHUNKS", 3)
    # Print "Speech detected..." style progress to the console.
    VERBOSE: bool = _env_bool("VAD_VERBOSE", True)


class TTSConfig:
    # "piper" (recommended) or "pyttsx3" (Windows SAPI5 fallback).
    ENGINE: str = _env_str("TTS_ENGINE", "piper")
    # Piper voice name — resolved relative to the voices/ directory.
    VOICE: str = _env_str("TTS_VOICE", "en_US-lessac-medium")
    # Optional explicit path to the ONNX voice model.
    VOICE_PATH: str = _env_str("TTS_VOICE_PATH", "")
    # Piper synthesis tuning.
    LENGTH_SCALE: float = _env_float("TTS_LENGTH_SCALE", 1.0)
    NOISE_SCALE: float = _env_float("TTS_NOISE_SCALE", 0.667)
    NOISE_W: float = _env_float("TTS_NOISE_W", 0.8)
    # pyttsx3 fallback settings (only used when TTS_ENGINE=pyttsx3).
    RATE: int = _env_int("TTS_RATE", 200)
    VOLUME: float = _env_float("TTS_VOLUME", 1.0)
    VOICE_INDEX: int = _env_int("TTS_VOICE_INDEX", 0)


class SearchConfig:
    # Web search provider: "tavily" | "serper" | "brave" | "" (disabled).
    PROVIDER: str = _env_str("SEARCH_PROVIDER", "tavily")
    # API key for the selected provider. Empty (or a placeholder value)
    # = web search disabled (current-information questions then answer
    # "cannot verify"). Placeholder keys are normalized to empty here so
    # they can never silently fail every call with an auth error.
    API_KEY: str = _env_secret("SEARCH_API_KEY")
    # How many results to fetch per search.
    MAX_RESULTS: int = _env_int("SEARCH_MAX_RESULTS", 5)
    # Seconds identical queries are served from cache (0 disables).
    CACHE_TTL: int = _env_int("SEARCH_CACHE_TTL", 300)
    # Upper bound on cached queries (oldest evicted first).
    CACHE_MAX_ENTRIES: int = _env_int("SEARCH_CACHE_MAX_ENTRIES", 50)
    # Sliding-window rate limit on external API calls (free tiers are
    # tiny: Tavily 1000/month). N calls are allowed per WINDOW seconds;
    # 0 disables the limiter.
    RATE_LIMIT: int = _env_int("SEARCH_RATE_LIMIT", 10)
    RATE_WINDOW: float = _env_float("SEARCH_RATE_WINDOW", 60.0)


class MemoryConfig:
    # Where the conversation history is persisted (JSON). Empty = in-memory only.
    FILE: str = _env_str("MEMORY_FILE", "data/conversation.json")
    # Save conversation history across restarts.
    PERSIST: bool = _env_bool("MEMORY_PERSIST", True)


class GroqConfig:
    # Optional fallback LLM provider. Empty key = Groq disabled and the
    # assistant runs on Ollama alone. The key is only ever read from the
    # environment / .env — never hardcoded, never logged. Placeholder
    # values are normalized to empty (see _env_secret).
    API_KEY: str = _env_secret("GROQ_API_KEY")
    MODEL: str = _env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
    TIMEOUT: int = _env_int("GROQ_TIMEOUT", 60)
    TEMPERATURE: float = _env_float("GROQ_TEMPERATURE", 0.7)
    MAX_TOKENS: int = _env_int("GROQ_MAX_TOKENS", 200)


class JARVISConfig:
    NAME: str = _env_str("JARVIS_NAME", "JARVIS")
    OWNER: str = _env_str("JARVIS_OWNER", "Sir")

    # Which AI provider to use ("ollama" primary, "groq" optional fallback).
    AI_PROVIDER: str = _env_str("AI_PROVIDER", "ollama")

    # Max characters accepted per user utterance. Longer input is
    # rejected politely instead of being routed to handlers or the LLM.
    MAX_INPUT_CHARS: int = _env_int("JARVIS_MAX_INPUT_CHARS", 500)

    # Seconds a CONFIRM-permission command (shutdown/restart/sleep) stays
    # pending before it expires and can never be executed. 0 = no timeout.
    CONFIRMATION_TIMEOUT: int = _env_int("CONFIRMATION_TIMEOUT", 30)

    # When true, destructive confirmations require the user to echo a
    # random nonce code (e.g. "say the code a3f2 to confirm") before
    # they execute — a stray "yes" from another process or an injected
    # stdin write can never authorize the action.
    CONFIRMATION_REQUIRE_TOKEN: bool = _env_bool(
        "CONFIRMATION_REQUIRE_TOKEN", False
    )

    # Answer mode for the question classifier:
    #   auto  — decide per question whether fresh info is needed (default)
    #   local — always answer from the local LLM
    #   web   — always search before answering
    AI_MODE: str = _env_str("AI_MODE", "auto").strip().lower()

    # Which Ollama model to use:
    #   quality — OLLAMA_QUALITY_MODEL (falls back to OLLAMA_MODEL)
    #   fast    — OLLAMA_FAST_MODEL     (falls back to OLLAMA_MODEL)
    # Defaults to "quality" so behavior is unchanged unless the user
    # opts into fast mode.
    MODEL_MODE: str = _env_str("JARVIS_MODEL_MODE", "quality").strip().lower()

    # How many user+assistant turns to keep (smaller = faster prompts).
    MEMORY_MAX_TURNS: int = _env_int("MEMORY_MAX_TURNS", 6)
    # Max characters of history sent per request (older turns dropped).
    MEMORY_MAX_CHARS: int = _env_int("MEMORY_MAX_CHARS", 3000)
    ENABLE_FAST_RESPONSES: bool = _env_bool("ENABLE_FAST_RESPONSES", True)
    ENABLE_WARMUP: bool = _env_bool("ENABLE_WARMUP", True)

    # Streaming STT: transcribe 3-second windows while the user is still
    # speaking and feed partial results to the router (early "stop"
    # detection, faster perceived response). Costs extra CPU on
    # CPU-only machines, so it defaults to off.
    STT_STREAM: bool = _env_bool("STT_STREAM", False)

    # Wake-word detection: when enabled (and the openwakeword package +
    # model are available) JARVIS only listens after "hey jarvis"
    # instead of always-on VAD. Falls back to always-on when the
    # optional dependency is missing.
    ENABLE_WAKE_WORD: bool = _env_bool("ENABLE_WAKE_WORD", False)
    WAKE_WORD: str = _env_str("JARVIS_WAKE_WORD", "hey jarvis").strip().lower()

    # Barge-in: monitor the microphone while JARVIS is speaking and
    # interrupt TTS the moment the user talks. Needs a quiet room / echo
    # handling, so it defaults to off.
    ENABLE_BARGE_IN: bool = _env_bool("ENABLE_BARGE_IN", False)


class LogConfig:
    # Console level: INFO by default, DEBUG with `jarvis --debug`.
    LEVEL: str = _env_str("LOG_LEVEL", "INFO")
    FILE: str = _env_str("LOG_FILE", "jarvis.log")


# ── Singleton instances used across all modules ───────────────
ollama_config = OllamaConfig()
whisper_config = WhisperConfig()
stt_config = STTConfig()
vad_config = VADConfig()
tts_config = TTSConfig()
jarvis_config = JARVISConfig()
log_config = LogConfig()
search_config = SearchConfig()
memory_config = MemoryConfig()
groq_config = GroqConfig()


# ── Configuration validation ─────────────────────────────────
# validate_config() audits every setting that matters and returns a
# list of problems: {"setting", "message", "fatal"}.
#
#   fatal=True  — genuinely invalid required configuration; JARVIS
#                 should refuse to start until it is fixed.
#   fatal=False — a warning; the affected feature degrades gracefully.
#
# Secret values are NEVER included in any message — only setting names.

_VALID_TTS_ENGINES = ("piper", "pyttsx3")
_VALID_AI_PROVIDERS = ("ollama", "groq")
_VALID_AI_MODES = ("auto", "local", "web")
_VALID_WHISPER_DEVICES = ("cpu", "cuda", "auto")
_VALID_WHISPER_COMPUTE = ("int8", "float16", "int8_float16")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_SEARCH_PROVIDERS = ("tavily", "serper", "brave", "none", "disabled", "off", "local")


def _problem(setting: str, message: str, fatal: bool = False) -> dict:
    return {"setting": setting, "message": message, "fatal": fatal}


def validate_config() -> list[dict]:
    """Audit the loaded configuration. Returns a list of problem dicts
    (never raises). See module docstring for the shape."""
    problems: list[dict] = []
    c = jarvis_config

    # ── AI provider ────────────────────────────────────────────
    if (c.AI_PROVIDER or "ollama").strip().lower() not in _VALID_AI_PROVIDERS:
        problems.append(_problem(
            "AI_PROVIDER",
            f"'{c.AI_PROVIDER}' is not supported (use ollama or groq).",
            fatal=True,
        ))
    if (c.AI_MODE or "auto") not in _VALID_AI_MODES:
        problems.append(_problem(
            "AI_MODE",
            f"'{c.AI_MODE}' is not supported (use auto, local or web).",
        ))
    if (c.MODEL_MODE or "quality") not in ("fast", "quality"):
        problems.append(_problem(
            "JARVIS_MODEL_MODE",
            f"'{c.MODEL_MODE}' is not supported (use fast or quality).",
        ))
    if c.MAX_INPUT_CHARS <= 0:
        problems.append(_problem(
            "JARVIS_MAX_INPUT_CHARS",
            "must be > 0 characters.",
            fatal=True,
        ))
    if c.CONFIRMATION_TIMEOUT < 0:
        problems.append(_problem(
            "CONFIRMATION_TIMEOUT",
            "must be >= 0 seconds (0 disables the timeout).",
            fatal=True,
        ))
    if not c.WAKE_WORD.strip():
        problems.append(_problem(
            "JARVIS_WAKE_WORD",
            "must not be empty when wake-word detection is enabled.",
        ))

    # ── Ollama ─────────────────────────────────────────────────
    o = ollama_config
    if not o.BASE_URL.startswith(("http://", "https://")):
        problems.append(_problem(
            "OLLAMA_BASE_URL",
            "must start with http:// or https://.",
            fatal=True,
        ))
    if not (o.MODEL or "").strip():
        problems.append(_problem(
            "OLLAMA_MODEL", "must not be empty.", fatal=True,
        ))
    if c.MODEL_MODE == "fast" and not (o.FAST_MODEL or "").strip():
        problems.append(_problem(
            "OLLAMA_FAST_MODEL",
            "JARVIS_MODEL_MODE=fast but OLLAMA_FAST_MODEL is empty — "
            "falling back to OLLAMA_MODEL.",
        ))
    if o.TIMEOUT <= 0:
        problems.append(_problem(
            "OLLAMA_TIMEOUT", "must be > 0 seconds.", fatal=True,
        ))
    if not 0.0 <= o.TEMPERATURE <= 1.0:
        problems.append(_problem(
            "OLLAMA_TEMPERATURE", "must be between 0.0 and 1.0.",
            fatal=True,
        ))
    if o.NUM_PREDICT <= 0:
        problems.append(_problem(
            "OLLAMA_NUM_PREDICT", "must be > 0 tokens.", fatal=True,
        ))
    if o.NUM_CTX <= 0:
        problems.append(_problem(
            "OLLAMA_NUM_CTX", "must be > 0 tokens.", fatal=True,
        ))
    if o.CIRCUIT_THRESHOLD < 1:
        problems.append(_problem(
            "OLLAMA_CIRCUIT_THRESHOLD", "must be >= 1 failure.",
            fatal=True,
        ))
    if o.CIRCUIT_RECOVERY <= 0:
        problems.append(_problem(
            "OLLAMA_CIRCUIT_RECOVERY", "must be > 0 seconds.",
            fatal=True,
        ))

    # ── Groq (optional) ────────────────────────────────────────
    g = groq_config
    if g.API_KEY and not (g.MODEL or "").strip():
        problems.append(_problem(
            "GROQ_MODEL", "must not be empty when GROQ_API_KEY is set.",
        ))
    if g.API_KEY and g.TIMEOUT <= 0:
        problems.append(_problem(
            "GROQ_TIMEOUT", "must be > 0 seconds.",
        ))

    # ── Whisper / STT ──────────────────────────────────────────
    w = whisper_config
    if not (w.MODEL or "").strip():
        problems.append(_problem(
            "WHISPER_MODEL", "must not be empty.", fatal=True,
        ))
    if w.DEVICE not in _VALID_WHISPER_DEVICES:
        problems.append(_problem(
            "WHISPER_DEVICE",
            f"'{w.DEVICE}' is not supported (use cpu, cuda or auto).",
            fatal=True,
        ))
    if w.COMPUTE_TYPE not in _VALID_WHISPER_COMPUTE:
        problems.append(_problem(
            "WHISPER_COMPUTE_TYPE",
            f"'{w.COMPUTE_TYPE}' is not supported "
            "(use int8, float16 or int8_float16).",
            fatal=True,
        ))
    if w.BEAM_SIZE < 1:
        problems.append(_problem(
            "WHISPER_BEAM_SIZE", "must be >= 1.",
        ))

    s = stt_config
    if not 8000 <= s.SAMPLE_RATE <= 48000:
        problems.append(_problem(
            "SAMPLE_RATE", "must be between 8000 and 48000 Hz.",
            fatal=True,
        ))
    if s.TIMEOUT <= 0:
        problems.append(_problem(
            "STT_TIMEOUT", "must be > 0 seconds.", fatal=True,
        ))
    if s.PHRASE_LIMIT <= 0:
        problems.append(_problem(
            "STT_PHRASE_LIMIT", "must be > 0 seconds.", fatal=True,
        ))
    if s.INPUT_DEVICE is not None and s.INPUT_DEVICE < 0:
        problems.append(_problem(
            "INPUT_DEVICE", "must be a non-negative device index.",
            fatal=True,
        ))

    # ── VAD ────────────────────────────────────────────────────
    v = vad_config
    if v.SILENCE_DURATION <= 0:
        problems.append(_problem(
            "VAD_SILENCE_DURATION", "must be > 0 seconds.", fatal=True,
        ))
    if v.CHUNK_DURATION <= 0:
        problems.append(_problem(
            "VAD_CHUNK_DURATION", "must be > 0 seconds.", fatal=True,
        ))
    if v.THRESHOLD_MULTIPLIER <= 0:
        problems.append(_problem(
            "VAD_THRESHOLD_MULTIPLIER", "must be > 0.", fatal=True,
        ))
    if v.MIN_THRESHOLD < 0:
        problems.append(_problem(
            "VAD_MIN_THRESHOLD", "must be >= 0.", fatal=True,
        ))
    if v.CALIBRATE_SECONDS < 0:
        problems.append(_problem(
            "VAD_CALIBRATE_SECONDS", "must be >= 0 seconds.",
        ))
    if v.MIN_SPEECH_CHUNKS < 1:
        problems.append(_problem(
            "VAD_MIN_SPEECH_CHUNKS", "must be >= 1.", fatal=True,
        ))

    # ── TTS ────────────────────────────────────────────────────
    t = tts_config
    if t.ENGINE not in _VALID_TTS_ENGINES:
        problems.append(_problem(
            "TTS_ENGINE",
            f"'{t.ENGINE}' is not supported (use piper or pyttsx3).",
            fatal=True,
        ))
    if t.ENGINE == "piper" and not (t.VOICE or "").strip():
        problems.append(_problem(
            "TTS_VOICE", "must not be empty when TTS_ENGINE=piper.",
        ))
    if t.VOICE_PATH and not os.path.isfile(t.VOICE_PATH):
        problems.append(_problem(
            "TTS_VOICE_PATH",
            f"points to a file that does not exist: {t.VOICE_PATH}",
        ))
    if t.ENGINE == "pyttsx3" and not 1 <= t.RATE <= 500:
        problems.append(_problem(
            "TTS_RATE", "must be between 1 and 500 words per minute.",
        ))
    if not 0.0 <= t.VOLUME <= 1.0:
        problems.append(_problem(
            "TTS_VOLUME", "must be between 0.0 and 1.0.",
        ))

    # ── Memory ─────────────────────────────────────────────────
    m = memory_config
    if c.MEMORY_MAX_TURNS < 1:
        problems.append(_problem(
            "MEMORY_MAX_TURNS", "must be >= 1.", fatal=True,
        ))
    if c.MEMORY_MAX_CHARS < 0:
        problems.append(_problem(
            "MEMORY_MAX_CHARS", "must be >= 0 (0 = unlimited).", fatal=True,
        ))
    if m.PERSIST and not m.FILE.strip():
        problems.append(_problem(
            "MEMORY_FILE", "must not be empty when MEMORY_PERSIST is on.",
        ))

    # ── Search (optional) ──────────────────────────────────────
    q = search_config
    if (q.PROVIDER or "").strip().lower() not in _VALID_SEARCH_PROVIDERS:
        problems.append(_problem(
            "SEARCH_PROVIDER",
            f"'{q.PROVIDER}' is unknown; web search will stay disabled.",
        ))

    # Placeholder API keys are normalized to "" (search disabled), but
    # the user still deserves to know their .env has a fake key.
    raw_search_key = (os.getenv("SEARCH_API_KEY", "") or "").strip()
    if raw_search_key and raw_search_key.lower() in _PLACEHOLDER_SECRETS:
        problems.append(_problem(
            "SEARCH_API_KEY",
            "looks like a placeholder value — web search is disabled "
            "until a real key is set. (The value itself is never logged.)",
        ))
    raw_groq_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if raw_groq_key and raw_groq_key.lower() in _PLACEHOLDER_SECRETS:
        problems.append(_problem(
            "GROQ_API_KEY",
            "looks like a placeholder value — Groq fallback is disabled "
            "until a real key is set.",
        ))
    if not 1 <= q.MAX_RESULTS <= 20:
        problems.append(_problem(
            "SEARCH_MAX_RESULTS", "must be between 1 and 20.",
        ))
    if q.CACHE_TTL < 0:
        problems.append(_problem(
            "SEARCH_CACHE_TTL", "must be >= 0 seconds (0 disables the cache).",
        ))
    if q.CACHE_MAX_ENTRIES < 1:
        problems.append(_problem(
            "SEARCH_CACHE_MAX_ENTRIES", "must be >= 1.",
            fatal=True,
        ))
    if q.RATE_LIMIT < 0:
        problems.append(_problem(
            "SEARCH_RATE_LIMIT", "must be >= 0 (0 disables rate limiting).",
            fatal=True,
        ))
    if q.RATE_WINDOW <= 0:
        problems.append(_problem(
            "SEARCH_RATE_WINDOW", "must be > 0 seconds.",
            fatal=True,
        ))

    # ── Logging ────────────────────────────────────────────────
    if (log_config.LEVEL or "INFO").strip().upper() not in _VALID_LOG_LEVELS:
        problems.append(_problem(
            "LOG_LEVEL",
            f"'{log_config.LEVEL}' is not a valid log level.",
        ))

    return problems
