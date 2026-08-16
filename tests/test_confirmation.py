"""Confirmation flow + web-search integration tests (all faked)."""

import commands.system_commands as sc
from brain.memory import ConversationMemory
from brain.router import IntentRouter
from brain.search import SearchResult
from commands.registry import CommandRegistry
from main import JARVIS


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def load(self):
        return True

    def speak(self, text):
        if text:
            self.spoken.append(text)

    def speak_blocking(self, text):
        self.speak(text)

    def wait(self):
        pass

    def stop(self):
        pass


class FakeSTT:
    vad = None

    def load(self):
        return True

    def listen(self):
        return None

    def unload(self):
        pass


class FakeMic:
    def is_available(self):
        return True

    def describe(self):
        return "fake mic"


class FakeProvider:
    name = "fake"

    def __init__(self, available=True, reply="A concise answer."):
        self.available = available
        self.reply = reply
        self.asked = []
        self.last_context = None

    def is_available(self):
        return self.available

    def ask(self, user_input, memory=None, context=None):
        self.asked.append(user_input)
        self.last_context = context
        return self.reply

    def ask_stream(self, user_input, memory=None, on_sentence=None, context=None):
        self.asked.append(user_input)
        self.last_context = context
        if on_sentence:
            on_sentence(self.reply)
        return self.reply

    def describe(self):
        return "fake provider"


class FakeSearch:
    name = "fake"

    def __init__(self, configured=True, results=None):
        self.configured = configured
        self.results = results if results is not None else []
        self.max_results = 5
        self.queries = []

    def is_configured(self):
        return self.configured

    def search(self, query, max_results=None):
        self.queries.append(query)
        return list(self.results)


def make_jarvis(provider=None, search=None, text_mode=True):
    return JARVIS(
        text_mode=text_mode,
        components={
            "mic": FakeMic(),
            "stt": FakeSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000),
            "provider": provider if provider is not None else FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
            "search": search if search is not None else FakeSearch(),
        },
    )


# ── CONFIRM permission flow ───────────────────────────────────

def test_shutdown_asks_for_confirmation_first(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    assert jarvis.process_input("shut down my computer") is True
    # Nothing executed yet — only the confirmation prompt was spoken.
    assert runs == []
    assert any("continue" in s.lower() for s in jarvis.tts.spoken)
    assert jarvis._pending is not None


def test_shutdown_executes_after_yes(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    jarvis.process_input("shut down my computer")
    jarvis.tts.spoken.clear()
    assert jarvis.process_input("yes") is True
    assert runs and runs[0][0] == "shutdown"
    assert "Shutting down" in jarvis.tts.spoken[-1]
    assert jarvis._pending is None


def test_shutdown_cancelled_on_no(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    jarvis.process_input("shut down my computer")
    jarvis.tts.spoken.clear()
    assert jarvis.process_input("no") is True
    assert runs == []
    assert any("won't" in s.lower() for s in jarvis.tts.spoken)
    assert jarvis._pending is None


def test_non_yes_no_drops_pending_and_continues(monkeypatch):
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))
    jarvis = make_jarvis()
    jarvis.process_input("shut down my computer")
    assert jarvis._pending is not None
    # A fresh question cancels the pending action without executing it.
    assert jarvis.process_input("what is python") is True
    assert runs == []
    assert jarvis._pending is None


def test_confirm_commands_never_run_via_registry_execute():
    jarvis = make_jarvis()
    # Back-compat entry point must also refuse to auto-execute.
    response = jarvis.commands.execute("shut down my computer")
    assert "continue" in response


# ── Stop speaking ─────────────────────────────────────────────

def test_stop_speaking_interrupts_tts():
    jarvis = make_jarvis()
    stopped = []

    jarvis.tts.stop = lambda: stopped.append(True)  # type: ignore[method-assign]
    assert jarvis.process_input("stop speaking") is True
    assert stopped


# ── Web search integration ────────────────────────────────────

RESULTS = [
    SearchResult(
        title="Andhra Pradesh CM",
        url="https://www.india.gov.in/cm",
        snippet="N. Chandrababu Naidu is the current Chief Minister.",
    ),
]


def test_web_search_uses_provider_context(monkeypatch, capsys):
    provider = FakeProvider(reply="The Chief Minister is N. Chandrababu Naidu.")
    search = FakeSearch(results=RESULTS)
    jarvis = make_jarvis(provider=provider, search=search)

    assert jarvis.process_input("who is the current chief minister of andhra pradesh") is True
    assert provider.asked == ["who is the current chief minister of andhra pradesh"]
    assert search.queries  # a search ran
    # The provider received the search results as context (never asks
    # the model the bare question without them).
    assert provider.last_context is not None
    assert "india.gov.in" in provider.last_context
    assert "Naidu" in provider.last_context
    assert "Naidu" in jarvis.tts.spoken[-1]
    out = capsys.readouterr().out
    assert "Searching..." in out
    assert "Sources" in out


def test_web_search_unconfigured_says_cannot_verify(monkeypatch, capsys):
    provider = FakeProvider()
    search = FakeSearch(configured=False)
    jarvis = make_jarvis(provider=provider, search=search)
    assert jarvis.process_input("who is the current chief minister") is True
    # No stale LLM answer — the honest cannot-verify message instead.
    assert "couldn't verify" in jarvis.tts.spoken[-1]
    assert provider.asked == []


def test_web_search_failure_says_cannot_verify():
    class BoomSearch(FakeSearch):
        def search(self, query, max_results=None):
            raise RuntimeError("network down")

    provider = FakeProvider(reply="stale answer from training")
    jarvis = make_jarvis(provider=provider, search=BoomSearch())
    assert jarvis.process_input("what happened today") is True
    assert "couldn't verify" in jarvis.tts.spoken[-1]
    assert provider.asked == []  # the stale answer was never used


def test_web_search_no_results_says_cannot_verify():
    provider = FakeProvider(reply="stale answer")
    jarvis = make_jarvis(provider=provider, search=FakeSearch(results=[]))
    assert jarvis.process_input("what is the latest news") is True
    assert "couldn't verify" in jarvis.tts.spoken[-1]
    assert provider.asked == []


def test_web_search_llm_down_reads_snippet():
    # LLM unavailable: answer straight from the verified top snippet.
    jarvis = make_jarvis(
        provider=FakeProvider(available=False),
        search=FakeSearch(results=RESULTS),
    )
    jarvis.ollama_ok = False
    assert jarvis.process_input("who is the current chief minister") is True
    assert any("Naidu" in s for s in jarvis.tts.spoken)


def test_web_search_streams_in_voice_mode():
    provider = FakeProvider(reply="It is N. Chandrababu Naidu.")
    jarvis = make_jarvis(
        provider=provider,
        search=FakeSearch(results=RESULTS),
        text_mode=False,
    )
    assert jarvis.process_input("who is the current chief minister") is True
    assert "Naidu" in jarvis.tts.spoken[-1]
    assert provider.last_context is not None


# ── Router routes current questions to WEB_SEARCH ─────────────

def test_router_returns_web_search_intent():
    from brain.router import Intent, IntentRouter

    router = IntentRouter()
    intent, _ = router.route("who is the current chief minister of andhra pradesh")
    assert intent == Intent.WEB_SEARCH

    intent, _ = router.route("what is python")
    assert intent == Intent.AI_QUESTION
