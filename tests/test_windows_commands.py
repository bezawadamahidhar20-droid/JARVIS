"""Windows command tests — the OS execution layer is fully mocked.

No application is ever launched, no screenshot is ever taken, and no
real shutdown command is issued during these tests.
"""

from pathlib import Path

import pytest

import commands.system_commands as sc
from commands.registry import (
    PERMISSION_CONFIRM,
    PERMISSION_SAFE,
    CommandRegistry,
)


@pytest.fixture()
def registry():
    return CommandRegistry()


# ── Folders ───────────────────────────────────────────────────

def test_open_downloads(registry, monkeypatch):
    opened = []
    monkeypatch.setattr(sc, "_folder_path", lambda key: Path("C:/fake/Downloads"))
    monkeypatch.setattr(sc, "_startfile", lambda p: opened.append(p))
    result = registry.execute("open downloads")
    assert "Opening Downloads" in result
    assert Path(opened[0]) == Path("C:/fake/Downloads")


def test_open_my_downloads_folder(registry, monkeypatch):
    opened = []
    monkeypatch.setattr(sc, "_folder_path", lambda key: Path("C:/fake/Downloads"))
    monkeypatch.setattr(sc, "_startfile", lambda p: opened.append(p))
    result = registry.execute("open my downloads folder")
    assert "Opening Downloads" in result
    assert Path(opened[0]) == Path("C:/fake/Downloads")


def test_open_documents(registry, monkeypatch):
    opened = []
    monkeypatch.setattr(sc, "_folder_path", lambda key: Path("C:/fake/Documents"))
    monkeypatch.setattr(sc, "_startfile", lambda p: opened.append(p))
    result = registry.execute("open documents")
    assert "Opening Documents" in result


def test_open_desktop(registry, monkeypatch):
    opened = []
    monkeypatch.setattr(sc, "_folder_path", lambda key: Path("C:/fake/Desktop"))
    monkeypatch.setattr(sc, "_startfile", lambda p: opened.append(p))
    result = registry.execute("open desktop")
    assert "Opening Desktop" in result


def test_folder_failure_is_friendly(registry, monkeypatch):
    monkeypatch.setattr(sc, "_folder_path", lambda key: Path("C:/fake/Downloads"))
    monkeypatch.setattr(sc, "_startfile", lambda p: (_ for _ in ()).throw(RuntimeError("denied")))
    result = registry.execute("open downloads")
    assert "couldn't open that folder" in result


# ── Apps ──────────────────────────────────────────────────────

def test_open_chrome_not_installed(registry, monkeypatch):
    launched = []

    def fake_popen(cmd):
        launched.append(cmd)
        raise FileNotFoundError("chrome.exe missing")

    monkeypatch.setattr(sc, "_chrome_path", lambda: r"C:\No\Chrome\chrome.exe")
    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)
    result = registry.execute("open chrome")
    assert "couldn't find chrome" in result or "not installed" in result


def test_open_settings_uses_uri(registry, monkeypatch):
    started = []
    monkeypatch.setattr(sc, "_startfile", lambda p: started.append(p))
    result = registry.execute("open settings")
    assert "Opening settings" in result
    assert started == ["ms-settings:"]


def test_open_powershell(registry, monkeypatch):
    launched = []
    monkeypatch.setattr(sc.subprocess, "Popen", lambda cmd: launched.append(cmd))
    result = registry.execute("open powershell")
    assert "Opening powershell" in result
    assert launched == ["powershell.exe"]


# ── System status ─────────────────────────────────────────────

def test_system_status_normal(registry, monkeypatch):
    monkeypatch.setattr(
        sc, "_system_metrics",
        lambda: {"cpu": 24.0, "ram": 48.0, "disk": 55.0},
    )
    result = registry.execute("system status")
    assert "running normally" in result
    assert "24" in result
    assert "48" in result


def test_system_status_heavy_load(registry, monkeypatch):
    monkeypatch.setattr(
        sc, "_system_metrics",
        lambda: {"cpu": 96.0, "ram": 88.0, "disk": 55.0},
    )
    result = registry.execute("how is my computer")
    assert "heavy load" in result


def test_system_status_disk_warning(registry, monkeypatch):
    monkeypatch.setattr(
        sc, "_system_metrics",
        lambda: {"cpu": 10.0, "ram": 20.0, "disk": 95.0},
    )
    result = registry.execute("computer status")
    assert "disk" in result and "95" in result


# ── Volume ────────────────────────────────────────────────────

class FakeVolumeController:
    def __init__(self):
        self.level = 0.5
        self.muted = False
        self.calls = []

    def GetMasterVolumeLevelScalar(self):
        return self.level

    def SetMasterVolumeLevelScalar(self, value, context=None):
        self.level = value
        self.calls.append(("set", value))

    def SetMute(self, mute, context=None):
        self.muted = bool(mute)
        self.calls.append(("mute", mute))


def test_volume_increase(registry, monkeypatch):
    ctrl = FakeVolumeController()
    monkeypatch.setattr(sc, "_volume_api", lambda: ctrl)
    result = registry.execute("increase volume")
    assert "increased" in result
    assert ctrl.level == 0.6


def test_volume_decrease(registry, monkeypatch):
    ctrl = FakeVolumeController()
    monkeypatch.setattr(sc, "_volume_api", lambda: ctrl)
    result = registry.execute("decrease volume")
    assert "decreased" in result
    assert ctrl.level == 0.4


def test_volume_mute_unmute(registry, monkeypatch):
    ctrl = FakeVolumeController()
    monkeypatch.setattr(sc, "_volume_api", lambda: ctrl)
    assert "muted" in registry.execute("mute volume")
    assert ctrl.muted is True
    assert "unmuted" in registry.execute("unmute volume")
    assert ctrl.muted is False


def test_volume_set_percent(registry, monkeypatch):
    ctrl = FakeVolumeController()
    monkeypatch.setattr(sc, "_volume_api", lambda: ctrl)
    result = registry.execute("set volume to 50 percent")
    assert "50" in result
    assert ctrl.level == 0.5


def test_volume_set_out_of_range(registry, monkeypatch):
    ctrl = FakeVolumeController()
    monkeypatch.setattr(sc, "_volume_api", lambda: ctrl)
    result = registry.execute("set volume to 150 percent")
    assert "between 0 and 100" in result
    assert ctrl.calls == []  # never executed


def test_volume_unavailable_reports_clearly(registry, monkeypatch):
    monkeypatch.setattr(sc, "_volume_api", lambda: None)
    result = registry.execute("increase volume")
    assert "pycaw" in result


# ── Lock screen ───────────────────────────────────────────────

def test_lock_screen(registry, monkeypatch):
    locked = []
    monkeypatch.setattr(sc, "_lock_workstation", lambda: locked.append(True) or True)
    result = registry.execute("lock my computer")
    assert "Locking" in result
    assert locked


def test_lock_screen_failure(registry, monkeypatch):
    monkeypatch.setattr(sc, "_lock_workstation", lambda: False)
    result = registry.execute("lock the computer")
    assert "couldn't lock" in result


# ── Power actions (CONFIRM) ───────────────────────────────────

def test_power_requires_confirmation(registry):
    """registry.execute() must never run a shutdown — it asks instead."""
    result = registry.execute_with_meta("shut down my computer")
    assert result is not None
    assert result.permission == PERMISSION_CONFIRM
    assert "continue" in result.confirm_prompt


def test_power_executes_only_via_result_execute(registry, monkeypatch):
    runs = []
    monkeypatch.setattr(
        sc.subprocess, "run", lambda args, **k: runs.append(args)
    )
    result = registry.execute_with_meta("restart my computer")
    response = result.execute("restart my computer")  # the confirmed step
    assert "Restarting" in response
    assert runs and runs[0][0] == "shutdown"


def test_sleep_is_confirm(registry):
    result = registry.execute_with_meta("sleep my computer")
    assert result.permission == PERMISSION_CONFIRM


def test_abort_shutdown_runs_immediately(registry, monkeypatch):
    runs = []
    monkeypatch.setattr(
        sc.subprocess, "run", lambda args, **k: runs.append(args)
    )
    result = registry.execute("abort shutdown")
    assert "Aborted" in result
    assert runs and runs[0][0] == "shutdown" and runs[0][1] == "/a"


def test_abort_shutdown_not_treated_as_power(registry, monkeypatch):
    """'abort shutdown' must never trigger a power-down."""
    runs = []
    monkeypatch.setattr(
        sc.subprocess, "run", lambda args, **k: runs.append(args)
    )
    result = registry.execute_with_meta("abort shutdown")
    assert result.permission == PERMISSION_SAFE
    assert runs and runs[0][1] == "/a"  # abort flag, never /s
    assert not any("/s" in r for r in runs)


# ── Screenshot ────────────────────────────────────────────────

def _patch_pyautogui(monkeypatch, captured):
    """pyautogui is imported lazily inside take_screenshot()."""
    import sys

    class FakePyautogui:
        @staticmethod
        def screenshot(path):
            captured.append(path)

    monkeypatch.setitem(sys.modules, "pyautogui", FakePyautogui)


def test_screenshot_saved_to_jarvis_folder(registry, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(sc.SystemCommands, "_screenshots_dir", lambda self: tmp_path)
    _patch_pyautogui(monkeypatch, captured)
    result = registry.execute("take a screenshot")
    assert "Saved a screenshot" in result
    assert captured and "screenshot_" in Path(captured[0]).name
    assert tmp_path == Path(captured[0]).parent


def test_screenshot_does_not_overwrite(registry, monkeypatch, tmp_path):
    from datetime import datetime as real_datetime

    class FixedNow:
        @classmethod
        def now(cls):
            return real_datetime(2026, 1, 1, 0, 0, 0, tzinfo=None)

    monkeypatch.setattr(sc, "datetime", FixedNow)
    # Pre-create a file with the exact stem the screenshot would use.
    (tmp_path / "screenshot_20260101_000000.png").write_bytes(b"old")
    captured = []
    monkeypatch.setattr(sc.SystemCommands, "_screenshots_dir", lambda self: tmp_path)
    _patch_pyautogui(monkeypatch, captured)
    result = registry.execute("take a screenshot")
    assert "Saved a screenshot" in result
    assert captured
    name = Path(captured[0]).name
    assert name != "screenshot_20260101_000000.png"  # never overwrites
    assert name.startswith("screenshot_20260101_000000_1")


def test_screenshot_missing_pyautogui(registry, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyautogui":
            raise ImportError("no pyautogui")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = registry.execute("take a screenshot")
    assert "isn't installed" in result


# ── Router ↔ registry vocabulary stays in sync ────────────────

def test_router_routes_new_commands(registry):
    from brain.router import Intent, IntentRouter

    router = IntentRouter()
    for phrase in (
        "open downloads", "open my documents", "system status",
        "increase volume", "mute volume", "set volume to 50 percent",
        "lock my computer", "shut down my computer", "abort shutdown",
        "open chrome", "open youtube", "take a screenshot",
    ):
        intent, cleaned = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"
        # And the registry must know what to do with it.
        assert registry.execute_with_meta(cleaned) is not None, phrase
