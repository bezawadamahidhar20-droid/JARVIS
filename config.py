"""
config.py — Central configuration for JARVIS
 
[FIX M1] All comparison operators corrected (<= instead of =).
         All error messages now match their actual constraints.
 
Loads all settings from .env file using python-dotenv.
All values use typed safe getters with fallbacks so a malformed
or missing .env value never crashes JARVIS.
"""
 
import os
from pathlib import Path
from dotenv import load_dotenv
 
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
 
 
_PLACEHOLDER_SECRETS = {
    "your-key-here", "your_key_here", "your key here", "changeme",
    "replace-me", "replace_me", "your_api_key_here", "your-api-key-here",
    "xxx", "sk-xxx", "add-your-key", "api-key", " ", "YOUR_API_KEY",
    "placeholder", "secret", "key", "12345",
}
 
 
def _env_secret(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        return ""
    if value.lower() in _PLACEHOLDER_SECRETS:
        return ""
    return value
 
 
class OllamaConfig:
    BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL: str = _env_str("OLLAMA_MODEL", "qwen3:8b")
    CIRCUIT_BREAKER: bool = _env_bool("OLLAMA_CIRCUIT_BREAKER", True)
    CIRCUIT_THRESHOLD: int = _env_int("OLLAMA_CIRCUIT_THRESHOLD", 3)
    CIRCUIT_RECOVERY: float = _env_float("OLLAMA_CIRCUIT_RECOVERY", 30.0)
    FAST_MODEL: str = _env_str("OLLAMA_FAST_MODEL", "")
    QUALITY_MODEL: str = _env_str("OLLAMA_QUALITY_MODEL", "")
    TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)
    TEMPERATURE: float = _env_float("OLLAMA_TEMPERATURE", 0.7)
    STREAM: bool = _env_bool("OLLAMA_STREAM", True)
    NUM_PREDICT: int = _env_int(
        "OLLAMA_NUM_PREDICT", _env_int("MAX_RESPONSE_TOKENS", 120),
    )
    NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 2048)
    KEEP_ALIVE: str = _env_str("OLLAMA_KEEP_ALIVE", "30m")
    NUM_GPU: int = _env_int("OLLAMA_NUM_GPU", 99)
    THINK: bool = _env_bool("OLLAMA_THINK", False)
 
    def resolve_model(self, mode: str | None = None) -> str:
        if mode is None:
            mode = (jarvis_config.MODEL_MODE or "quality").strip().lower()
        if mode == "fast" and (self.FAST_MODEL or "").strip():
            return self.FAST_MODEL.strip()
        if mode == "quality" and (self.QUALITY_MODEL or "").strip():
            return self.QUALITY_MODEL.strip()
        return (self.MODEL or "qwen3:8b").strip()
 
 
class WhisperConfig:
    MODEL: str = _env_str("WHISPER_MODEL", "base")
    COMPUTE_TYPE: str = _env_str("WHISPER_COMPUTE_TYPE", "int8")
    DEVICE: str = _env_str("WHISPER_DEVICE", "cpu")
    LANGUAGE: str = _env_str("WHISPER_LANGUAGE", "en")
    BEAM_SIZE: int = _env_int("WHISPER_BEAM_SIZE", 1)
    PRELOAD: bool = _env_bool("WHISPER_PRELOAD", True)
 
 
class STTConfig:
    SAMPLE_RATE: int = _env_int("SAMPLE_RATE", 16000)
    INPUT_DEVICE: int | None = (
        _env_int("INPUT_DEVICE", -1) if _env_int("INPUT_DEVICE", -1) >= 0 else None
    )
    LANGUAGE: str = _env_str("STT_LANGUAGE", "en-US")
    TIMEOUT: int = _env_int("STT_TIMEOUT", 5)
    PHRASE_LIMIT: int = _env_int("STT_PHRASE_LIMIT", 10)
 
 
class VADConfig:
    CALIBRATE_SECONDS: float = _env_float("VAD_CALIBRATE_SECONDS", 0.6)
    THRESHOLD_MULTIPLIER: float = _env_float("VAD_THRESHOLD_MULTIPLIER", 3.0)
    MIN_THRESHOLD: float = _env_float("VAD_MIN_THRESHOLD", 120.0)
    FIXED_THRESHOLD: float = _env_float("VAD_FIXED_THRESHOLD", 500.0)
    SILENCE_DURATION: float = _env_float("VAD_SILENCE_DURATION", 0.7)
    CHUNK_DURATION: float = _env_float("VAD_CHUNK_DURATION", 0.05)
    MIN_SPEECH_CHUNKS: int = _env_int("VAD_MIN_SPEECH_CHUNKS", 3)
    VERBOSE: bool = _env_bool("VAD_VERBOSE", True)
 
 
class TTSConfig:
    ENGINE: str = _env_str("TTS_ENGINE", "piper")
    VOICE: str = _env_str("TTS_VOICE", "en_US-lessac-medium")
    VOICE_PATH: str = _env_str("TTS_VOICE_PATH", "")
    LENGTH_SCALE: float = _env_float("TTS_LENGTH_SCALE", 1.0)
    NOISE_SCALE: float = _env_float("TTS_NOISE_SCALE", 0.667)
    NOISE_W: float = _env_float("TTS_NOISE_W", 0.8)
    RATE: int = _env_int("TTS_RATE", 200)
    VOLUME: float = _env_float("TTS_VOLUME", 1.0)
    VOICE_INDEX: int = _env_int("TTS_VOICE_INDEX", 0)
 
 
class SearchConfig:
    PROVIDER: str = _env_str("SEARCH_PROVIDER", "tavily")
    API_KEY: str = _env_secret("SEARCH_API_KEY")
    MAX_RESULTS: int = _env_int("SEARCH_MAX_RESULTS", 5)
    CACHE_TTL: int = _env_int("SEARCH_CACHE_TTL", 300)
    CACHE_MAX_ENTRIES: int = _env_int("SEARCH_CACHE_MAX_ENTRIES", 50)
    # [FIX m3] Rate limiting for search API calls
    RATE_LIMIT: int = _env_int("SEARCH_RATE_LIMIT", 10)
    RATE_WINDOW: int = _env_int("SEARCH_RATE_WINDOW", 60)
 
 
class MemoryConfig:
    FILE: str = _env_str("MEMORY_FILE", "data/conversation.json")
    PERSIST: bool = _env_bool("MEMORY_PERSIST", True)
 
 
class GroqConfig:
    API_KEY: str = _env_secret("GROQ_API_KEY")
    MODEL: str = _env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
    TIMEOUT: int = _env_int("GROQ_TIMEOUT", 60)
    TEMPERATURE: float = _env_float("GROQ_TEMPERATURE", 0.7)
    MAX_TOKENS: int = _env_int("GROQ_MAX_TOKENS", 200)
 
 
class JARVISConfig:
    NAME: str = _env_str("JARVIS_NAME", "JARVIS")
    WAKE_WORD: str = _env_str("JARVIS_WAKE_WORD", "jarvis")
    OWNER: str = _env_str("JARVIS_OWNER", "Sir")
    AI_PROVIDER: str = _env_str("AI_PROVIDER", "ollama")
    MAX_INPUT_CHARS: int = _env_int("JARVIS_MAX_INPUT_CHARS", 500)
    CONFIRMATION_TIMEOUT: int = _env_int("CONFIRMATION_TIMEOUT", 30)
    CONFIRMATION_REQUIRE_TOKEN: bool = _env_bool("CONFIRMATION_REQUIRE_TOKEN", False)
    AI_MODE: str = _env_str("AI_MODE", "auto").strip().lower()
    MODEL_MODE: str = _env_str("JARVIS_MODEL_MODE", "quality").strip().lower()
    MEMORY_MAX_TURNS: int = _env_int("MEMORY_MAX_TURNS", 6)
    MEMORY_MAX_CHARS: int = _env_int("MEMORY_MAX_CHARS", 3000)
    ENABLE_FAST_RESPONSES: bool = _env_bool("ENABLE_FAST_RESPONSES", True)
    ENABLE_WARMUP: bool = _env_bool("ENABLE_WARMUP", True)
    STT_STREAM: bool = _env_bool("STT_STREAM", False)
    ENABLE_WAKE_WORD: bool = _env_bool("ENABLE_WAKE_WORD", False)
    ENABLE_BARGE_IN: bool = _env_bool("ENABLE_BARGE_IN", False)
 
 
class LogConfig:
    LEVEL: str = _env_str("LOG_LEVEL", "INFO")
    FILE: str = _env_str("LOG_FILE", "jarvis.log")
 
 
# ── Singleton instances ───────────────────────────────────────
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
 
 
# ── Valid values ──────────────────────────────────────────────
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
    """
    [FIX M1] All comparison operators fixed (used <= instead of = assignment).
    All error messages now correctly describe their constraints.
    """
    problems: list[dict] = []
    c = jarvis_config
    o = ollama_config
    v = vad_config
    s = stt_config
    t = tts_config
    g = groq_config
    m = memory_config
    q = search_config
 
    # ── AI provider ──────────────────────────────────────────────
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
 
    # [FIX M1] Fixed: was "c.MAX_INPUT_CHARS = 0" (assignment), now "<= 0"
    if c.MAX_INPUT_CHARS <= 0:
        problems.append(_problem(
            "JARVIS_MAX_INPUT_CHARS",
            "must be > 0 characters.",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    # [FIX M1] Fixed: was "c.CONFIRMATION_TIMEOUT = 0", now "< 0"
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
 
    # ── Ollama ───────────────────────────────────────────────────
    if not o.BASE_URL.startswith(("http://", "https://")):
        problems.append(_problem(
            "OLLAMA_BASE_URL",
            "must start with http:// or https://.",
            fatal=True,
        ))
 
    if not (o.MODEL or "").strip():
        problems.append(_problem(
            "OLLAMA_MODEL",
            "must not be empty.",
            fatal=True,
        ))
 
    if c.MODEL_MODE == "fast" and not (o.FAST_MODEL or "").strip():
        problems.append(_problem(
            "OLLAMA_FAST_MODEL",
            "JARVIS_MODEL_MODE=fast but OLLAMA_FAST_MODEL is empty — "
            "falling back to OLLAMA_MODEL.",
        ))
 
    # [FIX M1] Fixed comparison operators
    if o.TIMEOUT <= 0:
        problems.append(_problem(
            "OLLAMA_TIMEOUT",
            "must be > 0 seconds.",
            fatal=True,
        ))
 
    if not 0.0 <= o.TEMPERATURE <= 2.0:
        problems.append(_problem(
            "OLLAMA_TEMPERATURE",
            "must be between 0.0 and 2.0.",
        ))
 
    if o.NUM_PREDICT <= 0:
        problems.append(_problem(
            "OLLAMA_NUM_PREDICT",
            "must be > 0 tokens.",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    if o.NUM_CTX <= 0:
        problems.append(_problem(
            "OLLAMA_NUM_CTX",
            "must be > 0 tokens.",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    if o.CIRCUIT_THRESHOLD < 1:
        problems.append(_problem(
            "OLLAMA_CIRCUIT_THRESHOLD",
            "must be >= 1 failure(s).",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    if o.CIRCUIT_RECOVERY <= 0:
        problems.append(_problem(
            "OLLAMA_CIRCUIT_RECOVERY",
            "must be > 0 seconds.",
            fatal=True,
        ))
 
    # ── Groq ─────────────────────────────────────────────────────
    if g.API_KEY and not (g.MODEL or "").strip():
        problems.append(_problem(
            "GROQ_MODEL",
            "must not be empty when GROQ_API_KEY is set.",
        ))
 
    if g.API_KEY and g.TIMEOUT <= 0:
        problems.append(_problem(
            "GROQ_TIMEOUT",
            "must be > 0 seconds.",
        ))
 
    # ── Whisper / STT ────────────────────────────────────────────
    w = whisper_config
    if not (w.MODEL or "").strip():
        problems.append(_problem(
            "WHISPER_MODEL",
            "must not be empty.",
            fatal=True,
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
            f"'{w.COMPUTE_TYPE}' is not supported (use int8, float16 or int8_float16).",
            fatal=True,
        ))
 
    if w.BEAM_SIZE < 1:
        problems.append(_problem(
            "WHISPER_BEAM_SIZE",
            "must be >= 1.",
        ))
 
    if not 8000 <= s.SAMPLE_RATE <= 48000:
        problems.append(_problem(
            "SAMPLE_RATE",
            "must be between 8000 and 48000.",
            fatal=True,
        ))
 
    if s.TIMEOUT <= 0:
        problems.append(_problem(
            "STT_TIMEOUT",
            "must be > 0 seconds.",
            fatal=True,
        ))
 
    if s.PHRASE_LIMIT <= 0:
        problems.append(_problem(
            "STT_PHRASE_LIMIT",
            "must be > 0 seconds.",
            fatal=True,
        ))
 
    if s.INPUT_DEVICE is not None and s.INPUT_DEVICE < 0:
        problems.append(_problem(
            "INPUT_DEVICE",
            "must be >= 0 (or unset for default).",
        ))
 
    # ── VAD ──────────────────────────────────────────────────────
    if v.SILENCE_DURATION <= 0:
        problems.append(_problem(
            "VAD_SILENCE_DURATION",
            "must be > 0 seconds.",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    if v.CHUNK_DURATION <= 0:
        problems.append(_problem(
            "VAD_CHUNK_DURATION",
            "must be > 0 seconds.",  # [FIX M1] Fixed message
            fatal=True,
        ))
 
    if v.THRESHOLD_MULTIPLIER <= 0:
        problems.append(_problem(
            "VAD_THRESHOLD_MULTIPLIER",
            "must be > 0.",
            fatal=True,
        ))
 
    if v.MIN_THRESHOLD < 0:
        problems.append(_problem(
            "VAD_MIN_THRESHOLD",
            "must be >= 0.",
            fatal=True,
        ))
 
    if v.CALIBRATE_SECONDS < 0:
        problems.append(_problem(
            "VAD_CALIBRATE_SECONDS",
            "must be >= 0 seconds.",
        ))
 
    if v.MIN_SPEECH_CHUNKS < 1:
        problems.append(_problem(
            "VAD_MIN_SPEECH_CHUNKS",
            "must be >= 1.",
            fatal=True,
        ))
 
    # ── TTS ──────────────────────────────────────────────────────
    if t.ENGINE not in _VALID_TTS_ENGINES:
        problems.append(_problem(
            "TTS_ENGINE",
            f"'{t.ENGINE}' is not supported (use piper or pyttsx3).",
            fatal=True,
        ))
 
    if t.ENGINE == "piper" and not (t.VOICE or "").strip():
        problems.append(_problem(
            "TTS_VOICE",
            "must not be empty when TTS_ENGINE=piper.",
        ))
 
    if t.VOICE_PATH and not os.path.isfile(t.VOICE_PATH):
        problems.append(_problem(
            "TTS_VOICE_PATH",
            f"points to a file that does not exist: {t.VOICE_PATH}",
        ))
 
    if t.ENGINE == "pyttsx3" and not 1 <= t.RATE <= 500:
        problems.append(_problem(
            "TTS_RATE",
            "must be between 1 and 500.",
        ))
 
    # ── Memory ───────────────────────────────────────────────────
    if c.MEMORY_MAX_TURNS < 1:
        problems.append(_problem(
            "MEMORY_MAX_TURNS",
            "must be >= 1.",
            fatal=True,
        ))
 
    if c.MEMORY_MAX_CHARS < 0:
        problems.append(_problem(
            "MEMORY_MAX_CHARS",
            "must be >= 0 (0 = unlimited).",
            fatal=True,
        ))
 
    if m.PERSIST and not m.FILE.strip():
        problems.append(_problem(
            "MEMORY_FILE",
            "must not be empty when MEMORY_PERSIST is on.",
        ))
 
    # ── Search ───────────────────────────────────────────────────
    if (q.PROVIDER or "").strip().lower() not in _VALID_SEARCH_PROVIDERS:
        problems.append(_problem(
            "SEARCH_PROVIDER",
            f"'{q.PROVIDER}' is unknown; web search will stay disabled.",
        ))
 
    raw_search_key = (os.getenv("SEARCH_API_KEY", "") or "").strip()
    if raw_search_key and raw_search_key.lower() in _PLACEHOLDER_SECRETS:
        problems.append(_problem(
            "SEARCH_API_KEY",
            "looks like a placeholder value — web search is disabled.",
        ))
 
    raw_groq_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if raw_groq_key and raw_groq_key.lower() in _PLACEHOLDER_SECRETS:
        problems.append(_problem(
            "GROQ_API_KEY",
            "looks like a placeholder value — Groq fallback is disabled.",
        ))
 
    if not 1 <= q.MAX_RESULTS <= 20:
        problems.append(_problem(
            "SEARCH_MAX_RESULTS",
            "must be between 1 and 20.",
        ))
 
    if q.CACHE_TTL < 0:
        problems.append(_problem(
            "SEARCH_CACHE_TTL",
            "must be >= 0 seconds (0 disables the cache).",
        ))
 
    if q.CACHE_MAX_ENTRIES < 1:
        problems.append(_problem(
            "SEARCH_CACHE_MAX_ENTRIES",
            "must be >= 1.",
        ))
 
    # ── Logging ──────────────────────────────────────────────────
    if (log_config.LEVEL or "INFO").strip().upper() not in _VALID_LOG_LEVELS:
        problems.append(_problem(
            "LOG_LEVEL",
            f"'{log_config.LEVEL}' is not a valid log level.",
        ))
 
    return problems
 