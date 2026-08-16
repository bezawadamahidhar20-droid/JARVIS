"""
jarvis_cli/startup.py — `jarvis --startup enable|disable`

Adds/removes a JARVIS shortcut in the current user's Windows Startup
folder so JARVIS launches automatically at login.

This is OPT-IN: nothing is added to startup unless the user runs
`jarvis --startup enable`. The command prints exactly what it changes
and how to undo it.

Uses pywin32 (installed with the project) to create a real .lnk;
falls back to PowerShell if pywin32 is missing.
"""

import os
import sys
import subprocess
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("startup")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _startup_folder() -> Path:
    """Current user's Startup folder (shell:startup)."""
    base = Path(os.environ.get(
        "APPDATA",
        str(Path.home() / "AppData" / "Roaming"),
    ))
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


SHORTCUT_NAME = "JARVIS.lnk"


def _shortcut_path() -> Path:
    return _startup_folder() / SHORTCUT_NAME


def _venv_pythonw() -> Path:
    """pythonw.exe inside the project venv (no console window)."""
    return PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"


def _create_shortcut_pywin32(target: str, args: str, workdir: str) -> None:
    import pythoncom
    from win32com.client import Dispatch

    pythoncom.CoInitialize()
    try:
        shell = Dispatch("WScript.Shell")
        lnk = shell.CreateShortCut(str(_shortcut_path()))
        lnk.TargetPath = target
        lnk.Arguments = args
        lnk.WorkingDirectory = workdir
        lnk.Description = "JARVIS — AI voice assistant (auto-start)"
        lnk.Save()
    finally:
        pythoncom.CoUninitialize()


def _create_shortcut_powershell(target: str, args: str, workdir: str) -> None:
    """Fallback when pywin32 isn't available."""
    ps = (
        f"$ws = New-Object -ComObject WScript.Shell; "
        f"$lnk = $ws.CreateShortcut('{_shortcut_path()}'); "
        f"$lnk.TargetPath = '{target}'; "
        f"$lnk.Arguments = '{args}'; "
        f"$lnk.WorkingDirectory = '{workdir}'; "
        f"$lnk.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=True,
        capture_output=True,
    )


def enable() -> int:
    """Install the startup shortcut. Prints what it changed."""
    target = str(_venv_pythonw())
    script = str(PROJECT_ROOT / "jarvis_cli" / "__main__.py")
    args = f'"{script}"'
    workdir = str(PROJECT_ROOT)

    if not Path(target).is_file():
        print(
            "[!] No venv found at "
            f"{PROJECT_ROOT / '.venv'} — run install.ps1 first."
        )
        return 1

    _startup_folder().mkdir(parents=True, exist_ok=True)
    try:
        try:
            _create_shortcut_pywin32(target, args, workdir)
        except ImportError:
            _create_shortcut_powershell(target, args, workdir)
    except Exception as e:
        logger.error(f"Could not create startup shortcut: {e}")
        print(f"[✗] Failed to enable startup: {e}")
        return 1

    print("[✓] JARVIS added to Windows startup.")
    print(f"    Shortcut: {_shortcut_path()}")
    print("    What changed: JARVIS now launches automatically when")
    print("    you log in, running in the background (voice mode,")
    print("    no console window).")
    print("    To undo: run  jarvis --startup disable")
    return 0


def disable() -> int:
    """Remove the startup shortcut. Prints what it changed."""
    path = _shortcut_path()
    if not path.exists():
        print("[i] JARVIS is not in Windows startup — nothing to remove.")
        return 0
    try:
        path.unlink()
    except Exception as e:
        logger.error(f"Could not remove startup shortcut: {e}")
        print(f"[✗] Failed to disable startup: {e}")
        return 1
    print("[✓] JARVIS removed from Windows startup.")
    print("    JARVIS will no longer start automatically at login.")
    return 0


def handle_startup(action: str) -> int:
    """Dispatch `jarvis --startup enable|disable`."""
    action = (action or "").lower()
    if action == "enable":
        return enable()
    if action == "disable":
        return disable()
    print(
        f"[!] Unknown --startup action '{action}'. "
        "Use: jarvis --startup enable | disable"
    )
    return 1


if __name__ == "__main__":
    sys.exit(handle_startup(sys.argv[1] if len(sys.argv) > 1 else ""))
