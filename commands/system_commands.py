"""System commands: open websites, launch apps, take screenshots.

Every external action is wrapped in try/except so a failure (missing app,
broken screenshot, unknown target) is reported as a friendly spoken message
instead of crashing JARVIS.
"""

import os
import re
import subprocess
import webbrowser

from utils.logger import get_logger

logger = get_logger("system_commands")

# ── Helpers (defined before the lookup tables that call them) ─────────────────

def _chrome_path() -> str:
    """Return the path to chrome.exe if installed in the usual spots."""
    candidates = (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return "chrome.exe"  # let subprocess fail loudly with a clear error


# ── Lookup tables ─────────────────────────────────────────────────────────────

# Spoken name -> website URL.
WEBSITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "mail": "https://mail.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "whatsapp": "https://web.whatsapp.com",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://maps.google.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.com",
    "linkedin": "https://www.linkedin.com",
}

# Spoken name -> how to launch it.
# A value ending in ".exe" is launched via PATH lookup (subprocess.Popen);
# anything else is treated as a shell URI for os.startfile (ms-settings: etc).
APPS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "terminal": "cmd.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "snipping tool": "snippingtool.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "edge": "msedge.exe",
    "browser": _chrome_path(),
    "chrome": _chrome_path(),
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "vscode": "code.exe",
    "visual studio code": "code.exe",
}

_SCREENSHOT_RE = re.compile(r"\bscreenshot\b", re.IGNORECASE)

# Capture the target after open/launch/start/go to/visit (optional "the").
_OPEN_RE = re.compile(
    r"\b(?:open|launch|start|go\s+to|visit)\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)


class SystemCommands:
    """Handles app/website launches and screenshots."""

    def execute(self, user_input: str) -> str:
        """Run the appropriate system action and return what to say."""
        t = (user_input or "").strip()
        if _SCREENSHOT_RE.search(t):
            return self.take_screenshot()
        return self._open_target(t)

    # ── Open website / app ────────────────────────────────────────────────────

    def _open_target(self, text: str) -> str:
        match = _OPEN_RE.search(text)
        if not match:
            return "I didn't catch what you wanted me to open."
        target = match.group(1).strip().lower()

        if target in WEBSITES:
            return self.open_website(WEBSITES[target])

        if target in APPS:
            return self.open_app(APPS[target], target)

        # Unknown name: if it looks like a domain, try it as a website.
        if re.fullmatch(r"[\w\-]+(\.[\w\-]+)+(/.*)?", target):
            url = target if target.startswith(("http://", "https://")) else f"https://{target}"
            return self.open_website(url)

        return f"I don't know how to open {target!r}."

    def open_website(self, url: str) -> str:
        """Open *url* in the default browser."""
        try:
            webbrowser.open(url, new=2)
            return f"Opening {url}."
        except Exception as exc:
            logger.error(f"Could not open website {url}: {exc}")
            return "Sorry, I couldn't open that website."

    def open_app(self, command: str, spoken_name: str) -> str:
        """Launch a Windows application."""
        try:
            if command.lower().endswith(".exe"):
                subprocess.Popen(command)
            else:
                os.startfile(command)  # type: ignore[attr-defined]  # e.g. ms-settings:
            return f"Opening {spoken_name}."
        except FileNotFoundError:
            logger.error(f"Application not found: {command}")
            return f"I couldn't find {spoken_name} on this machine."
        except Exception as exc:
            logger.error(f"Could not open {spoken_name}: {exc}")
            return f"Sorry, I couldn't open {spoken_name}."

    # ── Screenshot ───────────────────────────────────────────────────────────

    def take_screenshot(self) -> str:
        """Capture the whole screen with pyautogui into outputs/."""
        from datetime import datetime
        from pathlib import Path

        try:
            import pyautogui
        except ImportError:
            return "Screenshots need pyautogui, which isn't installed."

        try:
            out_dir = Path("outputs")
            out_dir.mkdir(exist_ok=True)
            filename = out_dir / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            pyautogui.screenshot(str(filename))
            return f"Saved a screenshot to {filename}."
        except Exception as exc:
            logger.error(f"Screenshot failed: {exc}")
            return "Sorry, I couldn't take a screenshot."