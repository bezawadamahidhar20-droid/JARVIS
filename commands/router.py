"""Safe command routing: local actions only ever come from fixed mappings.

Exit / quit detection is intentionally absent from this module.
It lives exclusively in ``main.py::is_exit_phrase()`` which checks the
*whole* utterance against a closed set of exit strings.

Keeping it out of the router prevents false-positives such as:
    "what is an exit code?"     → should answer, NOT shut down
    "how do I quit vim?"        → should answer, NOT shut down
    "tell me about exit polls"  → should answer, NOT shut down
"""

import os
import re
import subprocess
from datetime import datetime

# ── Pattern tables ────────────────────────────────────────────────────────────

TIME_PATTERNS = (
    "what time", "current time", "what's the time", "whats the time",
    "tell me the time", "time right now",
)

DATE_PATTERNS = (
    "what date", "current date", "what's the date", "whats the date",
    "tell me the date", "what day is it", "today's date",
)

CHROME_CANDIDATES = (
    os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Google", "Chrome", "Application", "chrome.exe",
    ),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_chrome() -> str | None:
    """Return the path to chrome.exe, or None if not found."""
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _launch_ui(path_or_args: str | list) -> None:
    """Launch a UI application without blocking the main thread."""
    if isinstance(path_or_args, str):
        os.startfile(path_or_args)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(list(path_or_args))


# ── Router ────────────────────────────────────────────────────────────────────

class CommandRouter:
    """Matches spoken text to a fixed set of safe local commands.

    Returns ``(command_name, None)`` when a command is matched so the caller
    can invoke ``execute()``.  Returns ``(None, None)`` to signal fall-through
    to the LLM.

    **Exit phrases are never matched here.**  They are handled by
    ``is_exit_phrase()`` in ``main.py``, which checks the *complete* utterance
    against a closed allow-list.  This is the design boundary that prevents
    mid-sentence words ("exit code", "quit vim") from shutting JARVIS down.
    """

    def route(self, text: str) -> tuple[str | None, None]:
        """Match *text* to a command name.

        The text is padded with spaces before comparison so substring matches
        cannot fire on partial words (e.g. ``"cmd"`` only matches ``\\bcmd\\b``).
        """
        t = " " + text.strip().lower() + " "

        # ── Clock / calendar ──────────────────────────────────────────────────
        if any(p in t for p in TIME_PATTERNS):
            return ("tell_time", None)
        if any(p in t for p in DATE_PATTERNS):
            return ("tell_date", None)

        # ── Applications ──────────────────────────────────────────────────────
        if "command prompt" in t or re.search(r"\bcmd\b", t):
            return ("open_cmd", None)
        if "file explorer" in t or re.search(r"\bexplorer\b", t):
            return ("open_explorer", None)
        if re.search(r"\bnotepad\b", t):
            return ("open_notepad", None)
        if re.search(r"\bcalculator\b|calc", t):
            return ("open_calculator", None)
        if re.search(r"\bchrome\b", t):
            return ("open_chrome", None)

        # ── No match → caller falls back to LLM ──────────────────────────────
        return (None, None)

    def execute(self, command: str) -> str:
        """Run a matched command and return what JARVIS should say.

        Raises
        ------
        ValueError
            If *command* is not a known command name (should never happen in
            normal operation because ``route()`` only returns valid names).
        RuntimeError
            If a required external resource (e.g. Chrome) is not available.
        """
        if command == "tell_time":
            now = datetime.now().strftime("%I:%M %p").lstrip("0")
            return f"The time is {now}."

        if command == "tell_date":
            today = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {today}."

        if command == "open_notepad":
            _launch_ui(["notepad.exe"])
            return "Opening Notepad."

        if command == "open_explorer":
            _launch_ui(["explorer.exe"])
            return "Opening File Explorer."

        if command == "open_cmd":
            subprocess.Popen(["cmd.exe"])
            return "Opening Command Prompt."

        if command == "open_calculator":
            try:
                os.startfile("calc:")  # type: ignore[attr-defined]
            except OSError:
                _launch_ui(["calc.exe"])
            return "Opening Calculator."

        if command == "open_chrome":
            chrome = _find_chrome()
            if chrome is None:
                raise RuntimeError("Chrome executable not found in standard paths.")
            _launch_ui([chrome])
            return "Opening Chrome."

        raise ValueError(f"Unknown command: {command!r}")
