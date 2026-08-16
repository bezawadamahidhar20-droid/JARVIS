"""Command registry tests — deterministic dispatch with no real side effects."""

import pytest

from commands.registry import CommandRegistry


@pytest.fixture()
def registry():
    return CommandRegistry()


def test_time_command(registry):
    result = registry.execute("what time is it")
    assert result.startswith("It's")


def test_date_command(registry):
    result = registry.execute("what's the date")
    assert result.startswith("Today is")


def test_open_website(registry, monkeypatch):
    import commands.system_commands as sc

    opened = []
    monkeypatch.setattr(sc.webbrowser, "open", lambda url, new: opened.append(url))
    result = registry.execute("open youtube")
    assert "Opening" in result
    assert opened == ["https://www.youtube.com"]


def test_open_app(registry, monkeypatch):
    import commands.system_commands as sc

    launched = []
    monkeypatch.setattr(
        sc.subprocess, "Popen", lambda cmd: launched.append(cmd)
    )
    result = registry.execute("open notepad")
    assert "Opening notepad" in result
    assert launched == ["notepad.exe"]


def test_unknown_command_polite_fallback(registry):
    result = registry.execute("open quantum-flux-device")
    assert "don't know how to open" in result


def test_screenshot(registry, monkeypatch):
    import sys

    class FakePyautogui:
        @staticmethod
        def screenshot(path):
            return None

    monkeypatch.setitem(sys.modules, "pyautogui", FakePyautogui)
    result = registry.execute("take a screenshot")
    assert "Saved a screenshot" in result


def test_screenshot_missing_pyautogui(registry, monkeypatch):
    # Force the ImportError branch by making the import fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyautogui":
            raise ImportError("no pyautogui")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = registry.execute("take a screenshot")
    assert "isn't installed" in result
