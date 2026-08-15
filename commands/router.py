"""Safe command routing: local actions only ever come from fixed mappings."""

import os
import re
import subprocess
from datetime import datetime

TIME_PATTERNS = (
    "what time", "current time", "what's the time", "whats the time",
    "tell me the time", "time right now",
)

DATE_PATTERNS = (
    "what date", "current date", "what's the date", "whats the date",
    "tell me the date", "what day is it", "today's date",
)

CHROME_CANDIDATES = (
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Google", "Chrome", "Application", "chrome.exe"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _launch_ui(path_or_args: str | list) -> None:
    if isinstance(path_or_args, str):
        os.startfile(path_or_args)
    else:
        subprocess.Popen(list(path_or_args))


class CommandRouter:
    """Matches spoken text to a fixed set of safe commands.

    Returns (name, response_text) if handled, otherwise (None, None) so
    the caller can fall back to Qwen3.
    """

    def route(self, text: str) -> tuple[str | None, None]:
        t = " " + text.strip().lower() + " "

        if any(p in t for p in TIME_PATTERNS):
            return ("tell_time", None)
        if any(p in t for p in DATE_PATTERNS):
            return ("tell_date", None)

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

        return (None, None)

    def execute(self, command: str) -> str:
        """Run a matched command and return what JARVIS should say.

        Exit/quit handling intentionally lives in main.py (is_exit_phrase),
        never here, so words like "exit" or "quit" inside a normal sentence
        can never trigger a shutdown.
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
                os.startfile("calc:")
            except OSError:
                _launch_ui(["calc.exe"])
            return "Opening Calculator."
        if command == "open_chrome":
            chrome = _find_chrome()
            if chrome is None:
                raise RuntimeError("Chrome executable not found in standard paths.")
            _launch_ui([chrome])
            return "Opening Chrome."
        raise ValueError(f"Unknown command: {command}")