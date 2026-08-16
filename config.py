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


class OllamaConfig:
    # Base URL of the local Ollama server.
    BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    # Model used as the AI brain (configurable; qwen3:8b is the default).
    MODEL: str = _env_str("OLLAMA_MODEL", "qwen3:8b")
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
    # API key for the selected provider. Empty = web search disabled
    # (current-information questions then answer "cannot verify").
    API_KEY: str = _env_str("SEARCH_API_KEY", "")
    # How many results to fetch per search.
    MAX_RESULTS: int = _env_int("SEARCH_MAX_RESULTS", 5)


class JARVISConfig:
    NAME: str = _env_str("JARVIS_NAME", "JARVIS")
    WAKE_WORD: str = _env_str("JARVIS_WAKE_WORD", "jarvis")
    OWNER: str = _env_str("JARVIS_OWNER", "Sir")

    # Which AI provider to use ("ollama" today; extensible).
    AI_PROVIDER: str = _env_str("AI_PROVIDER", "ollama")

    # Answer mode for the question classifier:
    #   auto  — decide per question whether fresh info is needed (default)
    #   local — always answer from the local LLM
    #   web   — always search before answering
    AI_MODE: str = _env_str("AI_MODE", "auto").strip().lower()

    # How many user+assistant turns to keep (smaller = faster prompts).
    MEMORY_MAX_TURNS: int = _env_int("MEMORY_MAX_TURNS", 6)
    # Max characters of history sent per request (older turns dropped).
    MEMORY_MAX_CHARS: int = _env_int("MEMORY_MAX_CHARS", 3000)
    ENABLE_FAST_RESPONSES: bool = _env_bool("ENABLE_FAST_RESPONSES", True)
    ENABLE_WARMUP: bool = _env_bool("ENABLE_WARMUP", True)


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
