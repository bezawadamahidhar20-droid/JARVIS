"""
jarvis_cli/doctor.py — `jarvis --doctor` health report

Checks every subsystem JARVIS depends on and prints a report like:

    [✓] Python 3.14.4
    [✓] Virtual environment  C:\\...\\.venv
    [✓] Microphone  Microphone Array (Intel Smart Sound)
    ...
    [✗] Ollama  not running
        Fix: start the Ollama app, or run `ollama serve`, then check
        http://localhost:11434

Doctor NEVER crashes: every check swallows its own exceptions and
reports the failure with a fix hint. Exits 0 when everything passes,
1 otherwise.
"""

import sys
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("doctor")

# Repository root (project root = parent of this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A check returns (ok: bool, detail: str).
_CHECK = tuple  # (name, fn, fix) — see _run_checks


def _check(name: str, fn, fix: str):
    """Decorator-free helper: build a check tuple."""
    return (name, fn, fix)


def _safe(fn):
    """Run *fn*, converting any exception into a failure result."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — doctor must never crash
        return False, str(e)


def _console_marks():
    """Return (ok, fail, warn) glyphs safe for the current console.

    Unicode ✓/✗ crash under the Windows cp1252 code page when stdout
    is piped (e.g. `jarvis --doctor | more`). Fall back to ASCII
    [OK]/[FAIL]/[WARN] there and keep the nicer glyphs on terminals
    that support them.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "\u2713".encode(encoding)
        "\u2717".encode(encoding)
        return ("[\u2713]", "[\u2717]", "[!]")
    except (UnicodeEncodeError, LookupError):
        return ("[OK]", "[FAIL]", "[WARN]")


# ── Individual checks ─────────────────────────────────────────

def _check_python():
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        return True, ver
    return False, ver


def _check_venv():
    venv_dir = PROJECT_ROOT / ".venv"
    if sys.prefix != sys.base_prefix:
        return True, sys.prefix
    if venv_dir.is_dir():
        return True, str(venv_dir)
    return False, "no virtual environment found"


def _check_microphone():
    import sounddevice as sd

    default_input = sd.default.device[0]
    if default_input is None:
        return False, "no default input device"
    info = sd.query_devices(default_input)
    return True, str(info.get("name", "Unknown"))


def _check_whisper():
    import faster_whisper  # noqa: F401

    from config import whisper_config

    return True, f"{whisper_config.MODEL} ({whisper_config.DEVICE}/{whisper_config.COMPUTE_TYPE})"


def _check_piper():
    import piper  # noqa: F401

    return True, "piper-tts installed"


def _check_voice_model():
    from config import tts_config

    if tts_config.VOICE_PATH:
        path = Path(tts_config.VOICE_PATH)
    else:
        path = PROJECT_ROOT / "voices" / f"{tts_config.VOICE}.onnx"
    if path.is_file():
        return True, str(path)
    return False, f"{path} not found"


def _check_ollama():
    from config import ollama_config
    import requests

    try:
        r = requests.get(
            f"{ollama_config.BASE_URL}/api/tags", timeout=3
        )
    except Exception:
        return False, f"{ollama_config.BASE_URL} not reachable"
    if r.status_code == 200:
        return True, ollama_config.BASE_URL
    return False, f"HTTP {r.status_code} from {ollama_config.BASE_URL}"


def _check_llm_model():
    from config import ollama_config
    import requests

    try:
        r = requests.get(
            f"{ollama_config.BASE_URL}/api/tags", timeout=3
        )
    except Exception:
        # Ollama down — the Ollama check reports it; not a model issue.
        return True, "Ollama not running (see Ollama check)"
    if r.status_code != 200:
        return True, "Ollama unreachable (see Ollama check)"
    models = [m.get("name", "") for m in r.json().get("models", [])]
    base = ollama_config.MODEL.split(":")[0]
    if any(base in n for n in models):
        return True, ollama_config.MODEL
    return False, (
        f"model '{ollama_config.MODEL}' not installed — "
        f"run: ollama pull {ollama_config.MODEL}"
    )


def _check_dirs():
    missing = []
    for name in ("voices", "outputs", "data"):
        d = PROJECT_ROOT / name
        if not d.is_dir():
            missing.append(name)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "voices/, outputs/, data/ present"


def _check_dependencies():
    missing = []
    for mod in (
        "numpy",
        "sounddevice",
        "faster_whisper",
        "piper",
        "requests",
        "dotenv",
        "pyautogui",
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "all core packages importable"


def _check_config():
    import config  # noqa: F401

    from config import (
        ollama_config,
        tts_config,
        whisper_config,
        vad_config,
    )

    problems = []
    if not ollama_config.BASE_URL.startswith(("http://", "https://")):
        problems.append("OLLAMA_BASE_URL invalid")
    if not whisper_config.MODEL:
        problems.append("WHISPER_MODEL empty")
    if tts_config.ENGINE not in ("piper", "pyttsx3"):
        problems.append(f"TTS_ENGINE '{tts_config.ENGINE}' unknown")
    if vad_config.SILENCE_DURATION <= 0:
        problems.append("VAD_SILENCE_DURATION must be > 0")
    if problems:
        return False, "; ".join(problems)
    detail = f".env at {PROJECT_ROOT / '.env'}"
    if not (PROJECT_ROOT / ".env").is_file():
        detail = "no .env (using built-in defaults)"
    return True, detail


def _check_web_search():
    """Informational: which search provider is selected?

    Never fails the doctor — web search is optional and JARVIS runs
    fine in LOCAL mode without it.
    """
    from config import search_config

    provider = (search_config.PROVIDER or "").strip().lower()
    if not provider or provider in ("none", "disabled", "off", "local"):
        return True, "not set (optional)"
    return True, provider.capitalize()


def _check_search_key():
    """Informational: is an API key set? Never prints the key itself."""
    from config import search_config

    if not search_config.API_KEY:
        return True, "not configured (optional)"
    return True, "configured"


def _check_search_reachable():
    """When configured, ping the search API once (doctor only).

    Distinguishes an invalid key (auth failure) from a temporary
    network problem so the user gets the right fix.
    """
    from config import search_config

    if not search_config.API_KEY:
        return True, "not configured (optional)"
    try:
        from brain.search import create_search_provider

        provider = create_search_provider()
        if provider is None:
            return False, "unknown SEARCH_PROVIDER"
        status = provider.is_reachable()
        name = provider.name.capitalize()
        if status == provider.REACH_OK:
            return True, f"{name} API reachable"
        if status == provider.REACH_AUTH:
            # Invalid key — a real problem, fail the doctor.
            return False, f"{name} authentication failed"
        if status == provider.REACH_NETWORK:
            # Transient problem — warn, but don't fail the doctor.
            return True, f"{name} temporarily unavailable", "warn"
        if status == provider.REACH_UNCONFIGURED:
            return True, "not configured (optional)"
        return False, f"{name} check failed (status: {status})"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"search check failed: {e}"


def _check_common_apps():
    """Detect common Windows apps. Informational — never fails."""
    from commands.system_commands import detect_common_apps

    try:
        found = detect_common_apps()
    except Exception as e:

        return True, f"detection failed: {e}"
    ok_mark, _, _ = _console_marks()
    parts = []
    for name in ("notepad", "calculator", "chrome", "edge", "explorer"):
        ok = found.get(name, False)
        parts.append(f"{name} {ok_mark if ok else 'not detected'}")
    return True, ", ".join(parts)


# ── Latency diagnostics ───────────────────────────────────────

def _check_ollama_streaming():
    """Streaming lets the first sentence be spoken while the rest of
    the reply is still generating."""
    from config import ollama_config

    if ollama_config.STREAM:
        return True, "enabled (sentence-by-sentence TTS)"
    return True, "disabled — waits for the full reply", "warn"


def _check_thinking():
    """Qwen3 emits hidden reasoning tokens unless think is disabled;
    on CPU that can add tens of seconds per reply."""
    from config import ollama_config

    if not ollama_config.THINK:
        return True, "disabled (recommended for voice)"
    return True, "enabled — adds reasoning latency", "warn"


def _check_model_keepalive():
    """Is keep_alive configured, and is the model actually in RAM?"""
    from config import ollama_config
    import requests

    if not ollama_config.KEEP_ALIVE:
        return True, "no keep_alive set — model reloads per question", "warn"
    try:
        r = requests.get(
            f"{ollama_config.BASE_URL}/api/ps", timeout=3
        )
    except Exception:
        # Ollama down — the Ollama check reports it; not a keep-alive issue.
        return True, "Ollama not running (see Ollama check)"
    if r.status_code != 200:
        return True, "cannot query Ollama", "warn"
    loaded = [m.get("name", "") for m in r.json().get("models", [])]
    base = ollama_config.MODEL.split(":")[0]
    if any(base in n for n in loaded):
        return True, (
            f"{ollama_config.MODEL} loaded (keep_alive "
            f"{ollama_config.KEEP_ALIVE})"
        )
    return True, (
        f"not loaded yet — loads on first question (keep_alive "
        f"{ollama_config.KEEP_ALIVE})"
    ), "warn"


# ── Report ────────────────────────────────────────────────────

CHECKS = [
    _check("Python", _check_python,
           "Install Python 3.10+ from https://www.python.org/downloads/ "
           "and enable 'Add python.exe to PATH'."),
    _check("Virtual environment", _check_venv,
           "Run:  python -m venv .venv  inside the repository, then "
           "install.ps1 or pip install -r requirements.txt."),
    _check("Microphone", _check_microphone,
           "Plug in a microphone and enable it in Windows Sound "
           "settings (Settings > System > Sound > Input)."),
    _check("Faster-Whisper", _check_whisper,
           "Run:  pip install faster-whisper"),
    _check("Piper", _check_piper,
           "Run:  pip install piper-tts"),
    _check("Voice model", _check_voice_model,
           "Download the voice with:  python -m piper.download_voices "
           "en_US-lessac-medium  (see TTS_VOICE in .env)"),
    _check("Ollama", _check_ollama,
           "Start the Ollama app, or run:  ollama serve\n"
           "Then check http://localhost:11434 in a browser."),
    _check("Selected LLM", _check_llm_model,
           "Run:  ollama pull <OLLAMA_MODEL>"),
    _check("Ollama streaming", _check_ollama_streaming,
           "Set OLLAMA_STREAM=true in .env for sentence-by-sentence "
           "speech instead of waiting for the whole reply."),
    _check("Thinking (reasoning)", _check_thinking,
           "Set OLLAMA_THINK=false in .env — Qwen3 otherwise spends "
           "tens of seconds on hidden reasoning tokens before speaking."),
    _check("Model kept alive", _check_model_keepalive,
           "Set OLLAMA_KEEP_ALIVE=30m in .env so the model stays in "
           "RAM between questions (no cold reload)."),
    _check("Required directories", _check_dirs,
           "Create the missing folders:  mkdir voices outputs data"),
    _check("Dependencies", _check_dependencies,
           "Run:  pip install -r requirements.txt"),
    _check("Configuration", _check_config,
           "Check .env values against .env.example"),
    _check("Web search provider", _check_web_search,
           "Set SEARCH_PROVIDER=tavily (or serper/brave) in .env to "
           "enable current-information answers (optional)."),
    _check("Search API key", _check_search_key,
           "Set SEARCH_API_KEY in .env to enable current-information "
           "answers (optional)."),
    _check("Search API", _check_search_reachable,
           "Check SEARCH_API_KEY / SEARCH_PROVIDER in .env."),
    _check("Common applications", _check_common_apps,
           "Install the missing apps; JARVIS still works without them."),
]


def run_doctor(verbose: bool = False) -> int:
    """Run all checks and print the report. Returns exit code."""
    print("")
    print("=============================================")
    print("          JARVIS DOCTOR — health report")
    print("=============================================")

    ok_mark, fail_mark, warn_mark = _console_marks()
    ok_count = 0
    fail_count = 0
    for name, fn, fix in CHECKS:
        result = _safe(fn)
        # A check may return a tri-state (ok, detail, status) where
        # status is "ok" / "fail" / "warn" — warn shows a [!]/[WARN]
        # mark without failing the doctor (e.g. search temporarily
        # unavailable, reasoning enabled).
        if isinstance(result, tuple) and len(result) >= 3:
            ok, detail, status = result[0], result[1], result[2]
            mark = {"ok": ok_mark, "fail": fail_mark, "warn": warn_mark}.get(
                status, warn_mark
            )
        else:
            ok, detail = result
            mark = ok_mark if ok else fail_mark
        print(f"  {mark} {name}  {detail}")
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            print(f"      Fix: {fix}")

    print("---------------------------------------------")
    if fail_count == 0:
        print(f"  All {ok_count} checks passed. JARVIS is ready.")
        print("  Type `jarvis` to start.")
        result = 0
    else:
        print(
            f"  {fail_count} check(s) failed, {ok_count} passed. "
            "See fixes above."
        )
        result = 1
    print("=============================================\n")

    if verbose:
        logger.debug(f"Doctor finished: {ok_count} ok, {fail_count} failed")
    return result


if __name__ == "__main__":
    sys.exit(run_doctor())
