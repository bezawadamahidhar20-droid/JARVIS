"""Central configuration for JARVIS.

Every setting is loaded from the ``.env`` file (or an environment variable
already set in the shell) using python-dotenv. Nothing secret or machine
specific is hardcoded here, so switching machines or tuning behaviour is a
matter of editing one file.

Rules of thumb:
* ``load_dotenv()`` reads ``.env`` from the current working directory.
* Values already present in the real environment win over ``.env``.
* If a key is missing or malformed we fall back to a sane default so the
  program never crashes at import time.
"""

import os

from dotenv import load_dotenv

# Load .env into the process environment (does nothing if .env is absent).
load_dotenv()


def _env_str(key: str, default: str) -> str:
    """Read a string setting, or *default* if unset."""
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    """Read an integer setting; a malformed value silently uses *default*.

    WHY: a typo in .env (e.g. "abc") must never crash JARVIS at startup.
    """
    raw = os.environ.get(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Read a float setting; a malformed value silently uses *default*."""
    raw = os.environ.get(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# ── Ollama / Qwen3 ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _env_str("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)         # seconds per request
OLLAMA_TEMPERATURE: float = _env_float("OLLAMA_TEMPERATURE", 0.7)

# ── Speech-to-text ────────────────────────────────────────────────────────────
STT_LANGUAGE: str = _env_str("STT_LANGUAGE", "en-US")
STT_TIMEOUT: int = _env_int("STT_TIMEOUT", 5)                 # wait for speech to start

# ── Text-to-speech ────────────────────────────────────────────────────────────
TTS_RATE: int = _env_int("TTS_RATE", 185)                     # words per minute

# ── Personality ───────────────────────────────────────────────────────────────
JARVIS_OWNER: str = _env_str("JARVIS_OWNER", "Sir")

# Number of full user+assistant turns kept for conversation context.
MEMORY_MAX_TURNS: int = _env_int("MEMORY_MAX_TURNS", 20)

SYSTEM_PROMPT: str = (
    "You are JARVIS, a witty British AI assistant. "
    "Answer any question naturally and conversationally. "
    f"Address the user as {JARVIS_OWNER}. "
    "Keep answers concise and write them in plain text — never use "
    "markdown symbols, bullet points or code blocks, because your reply "
    "is read aloud."
)