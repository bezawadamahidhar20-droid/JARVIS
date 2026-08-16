"""System commands: open websites, launch apps, open folders, screenshots,
system status, volume control, lock screen, and power actions.

Every external action is wrapped in try/except so a failure (missing app,
broken screenshot, unknown target) is reported as a friendly spoken message
instead of crashing JARVIS.

Security: every executable action is a *fixed, registered* handler. Nothing
here ever evaluates user text as a command — the target is resolved against
the safe lookup tables below (WEBSITES / APPS / FOLDERS).
"""

import difflib
import os
import re
import shutil
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from utils.logger import get_logger

logger = get_logger("system_commands")

try:
    from config import jarvis_config

    OWNER = jarvis_config.OWNER
except Exception:
    OWNER = "Sir"


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


def _find_exe(exe: str) -> bool:
    """True if *exe* is resolvable on PATH or in the usual install spots."""
    if shutil.which(exe):
        return True
    program_files = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for base in program_files:
        if not base:
            continue
        for root, _dirs, files in os.walk(base):
            if exe in files:
                return True
    return False


def _startfile(path: str) -> None:
    """Open a file/folder/URI with the OS default handler."""
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", path])


# ── Lookup tables ─────────────────────────────────────────────────────────────

# Spoken name -> website URL. Frozen (MappingProxyType) so no runtime
# code — nor anything importing this module — can mutate the trusted
# registry after import (command hijacking via registry mutation is
# impossible).
WEBSITES: MappingProxyType = MappingProxyType({
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
})

# Spoken name -> how to launch it. Frozen — see WEBSITES.
# A value ending in ".exe" is launched via PATH lookup (subprocess.Popen);
# anything else is treated as a shell URI for os.startfile (ms-settings: etc).
APPS: MappingProxyType = MappingProxyType({
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "terminal": "cmd.exe",
    "powershell": "powershell.exe",
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
    "vs code": "code.exe",
    "photos": "ms-photos:",
})

# ── Fuzzy matching (speech-recognition misspellings) ────────────────────────
# "open chrom" should resolve to Chrome; "open youtbe" to YouTube. We only
# ever match against the TRUSTED registries below (APPS / WEBSITES) with a
# high similarity cutoff — an arbitrary string is never executed.
_FUZZY_CUTOFF = 0.8
_FUZZY_MIN_LEN = 3


def fuzzy_match_target(target: str, registry: dict, cutoff: float = _FUZZY_CUTOFF) -> str | None:
    """Return the best trusted key for a possibly misspelled *target*.

    Only high-similarity matches against a known registry entry are
    returned — anything else is None (caller politely declines).
    """
    t = (target or "").strip().lower()
    if len(t) < _FUZZY_MIN_LEN:
        return None
    if not registry:
        return None
    matches = difflib.get_close_matches(t, registry.keys(), n=1, cutoff=cutoff)
    return matches[0] if matches else None

# Spoken name -> Windows known-folder key (resolved with SHGetKnownFolderPath).
# Frozen — see WEBSITES.
FOLDERS: MappingProxyType = MappingProxyType({
    "downloads": "downloads",
    "documents": "documents",
    "desktop": "desktop",
    "pictures": "pictures",
    "videos": "videos",
    "music": "music",
})

# Vocabulary exposed to the intent router so router and registry never drift.
APP_NAMES: tuple[str, ...] = tuple(APPS.keys())
SITE_NAMES: tuple[str, ...] = tuple(WEBSITES.keys())
FOLDER_NAMES: tuple[str, ...] = tuple(FOLDERS.keys())

_SCREENSHOT_RE = re.compile(r"\bscreenshot\b", re.IGNORECASE)

# Capture the target after open/launch/start/go to/visit (optional "the"/"my").
_OPEN_RE = re.compile(
    r"\b(?:open|launch|start|run|go\s+to|visit|show)\s+"
    r"(?:the\s+|my\s+)?(.+?)\s*$",
    re.IGNORECASE,
)


# ── Windows known folders ─────────────────────────────────────────────────────

_KNOWN_FOLDER_GUIDS = {
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
}

_FALLBACK_DIRS = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
}


def _known_folder_path(key: str) -> Path | None:
    """Resolve a Windows known folder (Downloads, Documents, ...) via
    SHGetKnownFolderPath. Returns None on any failure (e.g. non-Windows).
    """
    guid_str = _KNOWN_FOLDER_GUIDS.get(key)
    if not guid_str or os.name != "nt":
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        g = uuid.UUID(guid_str.strip("{}"))
        guid = wintypes.GUID()
        guid.Data1 = g.time_low
        guid.Data2 = g.time_mid
        guid.Data3 = g.time_hi_version
        guid.Data4 = (ctypes.c_ubyte * 8)(*g.bytes[8:])

        p = ctypes.c_wchar_p()
        hr = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(p)
        )
        if hr == 0 and p.value:
            path = str(p.value)
            ctypes.windll.ole32.CoTaskMemFree(p)
            return Path(path)
    except Exception as e:
        logger.debug(f"Known folder lookup failed for {key}: {e}")
    return None


def _folder_path(key: str) -> Path:
    """Best-effort path for a known folder (Windows API, then home dir)."""
    known = _known_folder_path(key)
    if known:
        return known
    return Path.home() / _FALLBACK_DIRS.get(key, key.capitalize())


# ── System status ─────────────────────────────────────────────────────────────

def _system_metrics() -> dict:
    """Return {cpu, ram, disk} percentages (0.0-100.0), never raising."""
    metrics = {"cpu": None, "ram": None, "disk": None}
    try:
        import psutil  # optional dependency

        metrics["cpu"] = psutil.cpu_percent(interval=0.3)
        metrics["ram"] = psutil.virtual_memory().percent
        metrics["disk"] = psutil.disk_usage(Path.home().anchor or "/").percent
        return metrics
    except Exception:
        pass

    # Fallbacks without psutil.
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):  # noqa: N801
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            metrics["ram"] = float(stat.dwMemoryLoad)
    except Exception:
        pass

    try:
        usage = shutil.disk_usage(Path.home().anchor or "/")
        metrics["disk"] = usage.used / usage.total * 100.0
    except Exception:
        pass

    try:
        import psutil  # noqa: F811 — re-import guard

        metrics["cpu"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    return metrics


# ── Volume control (optional pycaw dependency) ────────────────────────────────

def _volume_api():
    """Return a pycaw IAudioEndpointVolume controller, or None if the
    pycaw/comtypes libraries are unavailable."""
    try:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        logger.debug(f"Volume control unavailable (pycaw): {e}")
        return None


def _lock_workstation() -> bool:
    """Lock the Windows session. Returns True on success."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.LockWorkStation())
    except Exception as e:
        logger.error(f"Lock failed: {e}")
        return False


def _run(args: list[str]) -> bool:
    """Run an external process (no shell). Returns True on success."""
    try:
        subprocess.run(args, check=False, timeout=15)
        return True
    except FileNotFoundError:
        logger.error(f"Executable not found: {args[0]}")
        return False
    except Exception as e:
        logger.error(f"Command failed {args}: {e}")
        return False


# ── Common-app detection (for `jarvis --doctor`) ─────────────────────────────

def detect_common_apps() -> dict[str, bool]:
    """Check whether common Windows apps are present. Never raises."""
    result: dict[str, bool] = {}
    for name in ("notepad", "calculator", "chrome", "edge", "explorer"):
        exe = APPS.get(name, "")
        if exe.endswith(".exe"):
            result[name] = _find_exe(exe)
        else:
            result[name] = True  # shell URIs are always "available"
    return result


# ── Main command class ────────────────────────────────────────────────────────

class SystemCommands:
    """Handles app/website/folder launches, screenshots, system status,
    volume, lock, and power actions."""

    def execute(self, user_input: str) -> str:
        """Run the appropriate system action and return what to say."""
        t = (user_input or "").strip()
        if _SCREENSHOT_RE.search(t):
            return self.take_screenshot()
        if re.search(
            r"\b(system status|computer status|pc status|system info|"
            r"system information|system health|how is my computer|"
            r"how is the computer)\b", t, re.IGNORECASE,
        ):
            return self.system_status()
        if re.search(r"\bvolume\b|\b(mute|unmute)\b", t, re.IGNORECASE):
            return self.volume_control(t)
        if re.search(
            r"\block\s+(my\s+|the\s+)?(computer|pc|laptop|system|machine|screen)\b",
            t, re.IGNORECASE,
        ):
            return self.lock_screen()
        # Abort must be checked before the generic power pattern, otherwise
        # "abort shutdown" would match "shut*down" and power off the PC.
        if re.search(r"\b(abort|cancel)\s+shut\s*down\b", t, re.IGNORECASE):
            return self.abort_shutdown()
        if re.search(
            r"\b(shut\s*down|power off|restart|reboot|sleep|hibernate)\b",
            t, re.IGNORECASE,
        ):
            return self.power_action(t)
        return self._open_target(t)

    # ── Open website / app / folder ──────────────────────────────────────────

    def _open_target(self, text: str) -> str:
        match = _OPEN_RE.search(text)
        if not match:
            return "I didn't catch what you wanted me to open."
        target = match.group(1).strip().lower()

        if target in WEBSITES:
            return self.open_website(WEBSITES[target])

        if target in APPS:
            return self.open_app(APPS[target], target)

        # Speech recognition sometimes drops/transposes letters
        # ("open chrom" -> Chrome, "open youtbe" -> YouTube). Match only
        # against the trusted registries; never an arbitrary command.
        fuzzy = fuzzy_match_target(target, WEBSITES)
        if fuzzy:
            return self.open_website(WEBSITES[fuzzy])
        fuzzy = fuzzy_match_target(target, APPS)
        if fuzzy:
            return self.open_app(APPS[fuzzy], fuzzy)

        # Folders — strip "my"/"the" prefix and "folder" suffix first.
        key = re.sub(r"^(my|the)\s+", "", target)
        key = re.sub(r"\s+folder(s)?$", "", key).strip()
        if key in FOLDERS:
            return self.open_folder(_folder_path(key))

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
                # Absolute paths (e.g. chrome.exe resolved to its install
                # location) are checked up front so a missing binary gets
                # a clear "not installed" reply instead of a raw error.
                # Relative names (notepad.exe) are resolved via PATH and
                # only fail inside Popen.
                if os.path.isabs(command) and not os.path.isfile(command):
                    logger.error(
                        f"Application not found: {command}"
                    )
                    return (
                        f"I couldn't find {spoken_name} on this machine. "
                        "It may not be installed."
                    )
                subprocess.Popen(command)
            else:
                _startfile(command)  # e.g. ms-settings:, ms-photos:
            return f"Opening {spoken_name}."
        except FileNotFoundError:
            logger.error(f"Application not found: {command}")
            return (
                f"I couldn't find {spoken_name} on this machine. "
                "It may not be installed."
            )
        except Exception as exc:
            logger.error(f"Could not open {spoken_name}: {exc}")
            return f"Sorry, I couldn't open {spoken_name}."

    def open_folder(self, path: Path) -> str:
        """Open a folder in Windows Explorer."""
        try:
            _startfile(str(path))
            return f"Opening {path.name}."
        except Exception as exc:
            logger.error(f"Could not open folder {path}: {exc}")
            return "Sorry, I couldn't open that folder."

    # ── Screenshot ───────────────────────────────────────────────────────────

    def take_screenshot(self) -> str:
        """Capture the whole screen into Pictures/JARVIS/Screenshots/.

        Uses a timestamped filename so existing screenshots are never
        overwritten. Falls back to outputs/ if Pictures is unavailable.
        """
        try:
            import pyautogui
        except ImportError:
            return "Screenshots need pyautogui, which isn't installed."

        try:
            out_dir = self._screenshots_dir()
            stem = f"screenshot_{datetime.now():%Y%m%d_%H%M%S}"
            filename = self._unique_path(out_dir, stem)
            pyautogui.screenshot(str(filename))
            return f"Saved a screenshot to {filename}."
        except Exception as exc:
            logger.error(f"Screenshot failed: {exc}")
            return "Sorry, I couldn't take a screenshot."

    @staticmethod
    def _screenshots_dir() -> Path:
        """Pictures/JARVIS/Screenshots (created if missing), else outputs/."""
        try:
            base = Path.home() / "Pictures" / "JARVIS" / "Screenshots"
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception:
            base = Path("outputs")
            base.mkdir(exist_ok=True)
            return base

    @staticmethod
    def _unique_path(directory: Path, stem: str, ext: str = ".png") -> Path:
        """Return *directory/stem.png*, appending _1, _2, ... if taken."""
        candidate = directory / f"{stem}{ext}"
        i = 1
        while candidate.exists():
            candidate = directory / f"{stem}_{i}{ext}"
            i += 1
        return candidate

    # ── System status ────────────────────────────────────────────────────────

    def system_status(self) -> str:
        """Concise system health readout for the voice assistant."""
        m = _system_metrics()

        def _pct(value) -> str:
            return f"{value:.0f}" if value is not None else "?"

        cpu = _pct(m["cpu"])
        ram = _pct(m["ram"])
        disk = _pct(m["disk"])

        # Voice stays short: CPU + RAM, plus disk only if it is a concern.
        try:
            if (m["cpu"] is not None and m["cpu"] >= 85) or (
                m["ram"] is not None and m["ram"] >= 85
            ):
                return (
                    f"{OWNER}, your system is under heavy load. "
                    f"CPU usage is {cpu} percent and memory usage is "
                    f"{ram} percent."
                )
        except (TypeError, ValueError):
            pass

        if m["disk"] is not None and m["disk"] >= 90:
            return (
                f"{OWNER}, your system is running normally, but your disk "
                f"is {disk} percent full."
            )
        return (
            f"{OWNER}, your system is running normally. "
            f"CPU usage is {cpu} percent and memory usage is {ram} percent."
        )

    # ── Volume control ───────────────────────────────────────────────────────

    def volume_control(self, text: str) -> str:
        """Adjust the master volume (pycaw). Reports clearly when the
        optional audio library is unavailable instead of pretending."""
        api = _volume_api()
        if api is None:
            return (
                "Volume control isn't available — the pycaw audio library "
                "is not installed. Run: pip install pycaw comtypes"
            )

        # set volume to N percent (0-100)
        m = re.search(
            r"\bset\s+(?:the\s+)?volume\s+to\s+(\d{1,3})\s*%?\b",
            text, re.IGNORECASE,
        )
        if m:
            pct = int(m.group(1))
            if not 0 <= pct <= 100:
                return f"Volume must be between 0 and 100 percent, {OWNER}."
            try:
                api.SetMasterVolumeLevelScalar(pct / 100.0, None)
                return f"Volume set to {pct} percent."
            except Exception as exc:
                logger.error(f"Set volume failed: {exc}")
                return "Sorry, I couldn't set the volume."

        if re.search(r"\b(unmute|unmuted)\b", text, re.IGNORECASE):
            try:
                api.SetMute(0, None)
                return "Volume unmuted."
            except Exception as exc:
                logger.error(f"Unmute failed: {exc}")
                return "Sorry, I couldn't unmute the volume."

        if re.search(r"\bmute\b", text, re.IGNORECASE):
            try:
                api.SetMute(1, None)
                return "Volume muted."
            except Exception as exc:
                logger.error(f"Mute failed: {exc}")
                return "Sorry, I couldn't mute the volume."

        up = bool(
            re.search(r"\bvolume\s+up\b", text, re.IGNORECASE)
            or re.search(r"\b(increase|raise|turn\s+up)\b.*\bvolume\b", text, re.IGNORECASE)
        )
        down = bool(
            re.search(r"\bvolume\s+down\b", text, re.IGNORECASE)
            or re.search(r"\b(decrease|lower|turn\s+down)\b.*\bvolume\b", text, re.IGNORECASE)
        )
        try:
            current = float(api.GetMasterVolumeLevelScalar())
            if up:
                api.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
                return "Volume increased."
            if down:
                api.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)
                return "Volume decreased."
        except Exception as exc:
            logger.error(f"Volume step failed: {exc}")
            return "Sorry, I couldn't adjust the volume."

        return "I didn't catch the volume command."

    # ── Lock screen ──────────────────────────────────────────────────────────

    def lock_screen(self) -> str:
        """Lock the Windows workstation (exact recognized intent only)."""
        if _lock_workstation():
            return "Locking your screen, Sir."
        return "Sorry, I couldn't lock the screen."

    # ── Power actions (CONFIRM permission — run only after a yes) ────────────

    def power_action(self, text: str) -> str:
        """Execute shutdown / restart / sleep. Called only after the user
        confirms (see CommandRegistry permission handling)."""
        t = (text or "").lower()
        if re.search(r"\bshut\s*down\b|\bpower\s*off\b", t):
            if _run(["shutdown", "/s", "/t", "5"]):
                return f"Shutting down in 5 seconds, {OWNER}."
            return "Sorry, I couldn't shut down the computer."
        if re.search(r"\brestart\b|\breboot\b", t):
            if _run(["shutdown", "/r", "/t", "5"]):
                return f"Restarting in 5 seconds, {OWNER}."
            return "Sorry, I couldn't restart the computer."
        if re.search(r"\bsleep\b|\bhibernate\b", t):
            if _run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]):
                return f"Putting the system to sleep, {OWNER}."
            return "Sorry, I couldn't put the system to sleep."
        return "I don't know that power action."

    def abort_shutdown(self) -> str:
        """Cancel a scheduled shutdown/restart (shutdown /a)."""
        if _run(["shutdown", "/a"]):
            return "Aborted the scheduled shutdown."
        return "There was no scheduled shutdown to abort."
