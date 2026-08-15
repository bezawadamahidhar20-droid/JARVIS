"""
config.py — Central configuration for JARVIS
Loads all settings from .env file using python-dotenv.

All values use typed safe getters with fallbacks so a malformed or
missing .env value never crashes JARVIS.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# ── Typed getters with safe fallbacks ─────────────────────────
def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


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


class OllamaConfig:
    BASE_URL: str = _env_str(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    MODEL: str = _env_str(
        "OLLAMA_MODEL", "llama3.2:3b"
    )
    TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)
    TEMPERATURE: float = _env_float(
        "OLLAMA_TEMPERATURE", 0.7
    )
    STREAM: bool = _env_bool("OLLAMA_STREAM", True)
    # num_predict accepts the short alias MAX_RESPONSE_TOKENS too.
    NUM_PREDICT: int = _env_int(
        "OLLAMA_NUM_PREDICT",
        _env_int("MAX_RESPONSE_TOKENS", 150),
    )
    NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 2048)
    KEEP_ALIVE: str = _env_str("OLLAMA_KEEP_ALIVE", "30m")
    NUM_GPU: int = _env_int("OLLAMA_NUM_GPU", 99)


class STTConfig:
    ENGINE: str = _env_str("STT_ENGINE", "google")
    LANGUAGE: str = _env_str("STT_LANGUAGE", "en-US")
    TIMEOUT: int = _env_int("STT_TIMEOUT", 5)
    PHRASE_LIMIT: int = _env_int("STT_PHRASE_LIMIT", 10)

    # Seconds of silence that ends a phrase (0.7 = fast cut-off).
    SILENCE_DURATION: float = _env_float(
        "STT_SILENCE_DURATION", 0.7
    )
    # Audio chunk length in seconds (0.05 = snappy onset detection).
    CHUNK_DURATION: float = _env_float(
        "STT_CHUNK_DURATION", 0.05
    )
    # Google speech API key (kept out of source code).
    GOOGLE_KEY: str = _env_str(
        "GOOGLE_STT_KEY",
        "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw",
    )


class TTSConfig:
    ENGINE: str = _env_str("TTS_ENGINE", "pyttsx3")
    RATE: int = _env_int("TTS_RATE", 200)
    VOLUME: float = _env_float("TTS_VOLUME", 1.0)
    VOICE_INDEX: int = _env_int("TTS_VOICE_INDEX", 0)


class JARVISConfig:
    NAME: str = _env_str("JARVIS_NAME", "JARVIS")
    WAKE_WORD: str = _env_str("JARVIS_WAKE_WORD", "jarvis")
    OWNER: str = _env_str("JARVIS_OWNER", "Sir")

    # How many user+assistant turns to keep (smaller = faster prompts).
    MEMORY_MAX_TURNS: int = _env_int("MEMORY_MAX_TURNS", 6)
    # Max characters of history sent per request (older turns dropped).
    MEMORY_MAX_CHARS: int = _env_int("MEMORY_MAX_CHARS", 3000)
    ENABLE_FAST_RESPONSES: bool = _env_bool(
        "ENABLE_FAST_RESPONSES", True
    )
    ENABLE_WARMUP: bool = _env_bool("ENABLE_WARMUP", True)


class LogConfig:
    LEVEL: str = _env_str("LOG_LEVEL", "INFO")
    FILE: str = _env_str("LOG_FILE", "jarvis.log")


# ── Singleton instances used across all modules ───────────────
ollama_config = OllamaConfig()
stt_config = STTConfig()
tts_config = TTSConfig()
jarvis_config = JARVISConfig()
log_config = LogConfig()