"""Issue 10 — safe fuzzy application/website matching.

Speech-recognition misspellings ("open chrom") must resolve to the
trusted registry entry, while arbitrary lookalike strings must never
execute anything.
"""

import commands.system_commands as sc
from brain.router import Intent, IntentRouter
from commands.registry import CommandRegistry


def _open_with(monkeypatch, launched, started, webbrowser_opened):
    monkeypatch.setattr(sc.subprocess, "Popen",
                        lambda cmd: launched.append(cmd))
    monkeypatch.setattr(sc, "_startfile", lambda p: started.append(p))
    monkeypatch.setattr(sc.webbrowser, "open",
                        lambda url, new=2: webbrowser_opened.append(url))
    return CommandRegistry()


# ── "open chrom" -> Chrome ────────────────────────────────────

def test_open_chrom_fuzzy_resolves_to_chrome(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open chrom")
    assert "Opening chrome" in result
    # The trusted registry entry for chrome was launched — nothing else.
    assert launched == [sc.APPS["chrome"]]


def test_open_chrom_routes_as_command():
    router = IntentRouter()
    intent, _ = router.route("open chrom")
    assert intent == Intent.AI_QUESTION


# ── "open vs code" / "open vscode" -> VS Code ─────────────────

def test_open_vs_code_resolves_to_vscode(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open vs code")
    assert "Opening vs code" in result
    assert launched == ["code.exe"]


def test_open_vscode_resolves_to_vscode(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open vscode")
    assert "Opening vscode" in result
    assert launched == ["code.exe"]


def test_open_vs_code_routes_as_command():
    router = IntentRouter()
    intent, _ = router.route("open vs code")
    assert intent == Intent.AI_QUESTION


# ── Fuzzy websites ────────────────────────────────────────────

def test_open_youtbe_fuzzy_resolves_to_youtube(monkeypatch):
    opened = []
    registry = _open_with(monkeypatch, [], [], opened)
    result = registry.execute("open youtbe")
    assert "Opening https://www.youtube.com" in result
    assert opened == ["https://www.youtube.com"]


# ── Safety: never execute lookalikes ──────────────────────────

def test_unknown_target_declined_not_executed(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open quantumflux-device")
    assert "don't know how to open" in result
    assert launched == []


def test_low_similarity_target_declined(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open zzzznonexistent")
    assert "don't know how to open" in result
    assert launched == []


def test_short_target_never_fuzzy_matched(monkeypatch):
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open xy")
    assert "don't know how to open" in result
    assert launched == []


def test_fuzzy_match_target_respects_cutoff():
    # The helper itself: high-similarity only.
    assert sc.fuzzy_match_target("chrom", sc.APPS) == "chrome"
    assert sc.fuzzy_match_target("vs code", sc.APPS) == "vs code"
    assert sc.fuzzy_match_target("totally-unrelated", sc.APPS) is None
    assert sc.fuzzy_match_target("ab", sc.APPS) is None  # too short


def test_fuzzy_never_matches_arbitrary_commands(monkeypatch):
    """A malicious-looking target must not resolve to an app that runs
    shell commands."""
    launched = []
    registry = _open_with(monkeypatch, launched, [], [])
    result = registry.execute("open ; rm -rf /")
    assert "don't know how to open" in result
    assert launched == []
