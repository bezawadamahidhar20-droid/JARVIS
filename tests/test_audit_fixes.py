"""Audit-fix tests: log rotation, memory thread-safety, confirmation
nonce, placeholder secrets, frozen command tables, circuit breaker,
Ollama breaker wiring, and benchmark baseline comparison.
"""

import logging
import threading
import time
from logging.handlers import RotatingFileHandler

import pytest

import brain.ollama_client as oc
import commands.system_commands as sc
from brain.circuit_breaker import CircuitBreaker
from brain.exceptions import CircuitOpenError
from brain.memory import ConversationMemory
from brain.ollama_client import OllamaClient
from commands.registry import PendingConfirmation
from jarvis_cli.benchmark import (
    REGRESSION_RATIO,
    compare_baseline,
    save_baseline,
)


# ── Log rotation ──────────────────────────────────────────────

def test_logger_uses_rotating_file_handler():
    """jarvis.log must be written through a RotatingFileHandler so a
    long-running install never exhausts the disk."""
    from utils.logger import get_logger

    get_logger("audit_fixes_test")  # ensures root is configured
    root = logging.getLogger()
    handlers = [
        h for h in root.handlers
        if isinstance(h, RotatingFileHandler)
    ]
    assert handlers, "expected a RotatingFileHandler on the root logger"
    handler = handlers[0]
    assert handler.maxBytes == 10 * 1024 * 1024
    assert handler.backupCount == 5


# ── Memory thread-safety ──────────────────────────────────────

def test_memory_pop_last():
    m = ConversationMemory(max_turns=6, max_chars=3000, persist_path="")
    m.add_user_message("hello")
    m.add_assistant_message("hi there")
    assert m.pop_last() == {"role": "assistant", "content": "hi there"}
    assert m.pop_last() == {"role": "user", "content": "hello"}
    assert m.pop_last() is None  # empty -> None, never raises


def test_memory_is_thread_safe():
    """Concurrent add/pop from many threads must never corrupt state."""
    m = ConversationMemory(max_turns=6, max_chars=3000, persist_path="")
    errors = []

    def writer(n):
        try:
            for i in range(200):
                m.add_user_message(f"user-{n}-{i}")
                m.add_assistant_message(f"asst-{n}-{i}")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                m.get_history()
                m.pop_last()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(i,))
        for i in range(4)
    ] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # List invariant: never more than max_turns*2 messages.
    assert len(m) <= 6 * 2
    assert m.get_history() == list(m.get_history())


# ── Confirmation nonce ────────────────────────────────────────

class _FakeResult:
    name = "power"
    permission = "confirm"
    confirm_prompt = "That will close your session. Do you want me to continue?"

    def execute(self, text):
        return "done"


def _pending(**kw):
    return PendingConfirmation(_FakeResult(), "shut down my computer", **kw)


def test_confirmation_has_random_nonce():
    a = _pending()
    b = _pending()
    assert a.token and len(a.token) >= 6
    assert a.token != b.token  # every confirmation gets a fresh nonce


def test_confirmation_token_required_when_enabled():
    p = _pending(require_token=True)
    # A plain "yes" without the code is NOT authorized.
    assert p.confirm("yes") is False
    assert p.confirm("yes", "wrong-code") is False
    # Echoing the code authorizes (case-insensitive) — even when the
    # phrase is not parsed as a plain "yes" (the code IS the "yes").
    assert p.confirm("yes", p.token) is True
    assert p.confirm("yes", p.token.upper()) is True
    assert p.confirm("other", p.token) is True
    # An explicit "no" still cancels, code or not.
    assert p.confirm("no", p.token) is False


def test_confirmation_token_optional_by_default():
    p = _pending(require_token=False)
    assert p.confirm("yes") is True
    assert p.confirm("no") is False
    assert p.confirm("other") is False


def test_confirmation_prompt_includes_token_when_required():
    p = _pending(require_token=True)
    assert p.token in p.prompt
    p2 = _pending(require_token=False)
    assert p2.token not in p2.prompt


def test_confirmation_token_cannot_reauthorize_after_take():
    p = _pending(require_token=True)
    assert p.confirm("yes", p.token) is True
    assert p.take() is not None
    assert p.confirm("yes", p.token) is False  # consumed


def test_main_token_confirmation_flow(monkeypatch):
    """End-to-end: a token-gated confirmation executes only when the
    code is echoed back."""
    from brain.memory import ConversationMemory
    from brain.router import IntentRouter
    from commands.registry import CommandRegistry
    from main import JARVIS

    # Force token-gated confirmations for this test (the .env default
    # is off to preserve the plain "yes" UX).
    import commands.registry as registry_mod

    monkeypatch.setattr(registry_mod, "CONFIRMATION_REQUIRE_TOKEN", True)
    runs = []
    monkeypatch.setattr(sc.subprocess, "run", lambda args, **k: runs.append(args))

    class FakeTTS:
        spoken = []

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
            return "fake"

    class FakeProvider:
        name = "fake"

        def is_available(self):
            return True

        def ask(self, *_a, **_k):
            return None

        def ask_stream(self, *_a, **_k):
            return None

        def describe(self):
            return "fake"

    jarvis = JARVIS(
        text_mode=True,
        components={
            "mic": FakeMic(),
            "stt": FakeSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000, persist_path=""),
            "provider": FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
        },
    )

    # Request the shutdown -> confirmation pending with token required.
    assert jarvis.process_input("shut down my computer") is True
    assert jarvis._pending is not None and jarvis._pending.require_token
    token = jarvis._pending.token
    assert runs == []

    # Plain "yes" without the code: NOT authorized, pending dropped.
    assert jarvis.process_input("yes") is True
    assert runs == []
    assert jarvis._pending is None

    # Re-request and echo the NEW code -> executes exactly once. (Each
    # confirmation gets a fresh nonce, so the old token is now invalid.)
    assert jarvis.process_input("shut down my computer") is True
    fresh_token = jarvis._pending.token
    assert fresh_token != token  # nonce rotates per confirmation
    assert jarvis.process_input(fresh_token) is True
    assert runs and runs[0][0] == "shutdown"
    assert jarvis._pending is None


# ── Placeholder secrets ───────────────────────────────────────

def test_env_secret_normalizes_placeholders(monkeypatch):
    from config import _env_secret

    for placeholder in ("your-key-here", "changeme", "YOUR_KEY_HERE", "sk-xxx"):
        monkeypatch.setenv("JARVIS_TEST_KEY", placeholder)
        assert _env_secret("JARVIS_TEST_KEY") == "", placeholder

    monkeypatch.setenv("JARVIS_TEST_KEY", "tavily-real-key-abc123")
    assert _env_secret("JARVIS_TEST_KEY") == "tavily-real-key-abc123"

    monkeypatch.setenv("JARVIS_TEST_KEY", "   ")
    assert _env_secret("JARVIS_TEST_KEY") == ""


def test_validate_config_warns_on_placeholder_key(monkeypatch):
    from config import validate_config

    monkeypatch.setenv("SEARCH_API_KEY", "your-key-here")
    problems = validate_config()
    matching = [p for p in problems if p["setting"] == "SEARCH_API_KEY"]
    assert matching and matching[0]["fatal"] is False
    # The message must never contain the key value itself.
    assert "your-key-here" not in matching[0]["message"]


# ── Frozen command tables ─────────────────────────────────────

def test_command_tables_are_frozen():
    for table in (sc.WEBSITES, sc.APPS, sc.FOLDERS):
        with pytest.raises((TypeError, AttributeError)):
            table["hijacked"] = "malicious"
        with pytest.raises((TypeError, AttributeError)):
            table.clear()
    # Lookups still work through the read-only mapping.
    assert sc.APPS["notepad"] == "notepad.exe"
    assert "youtube" in sc.WEBSITES
    assert "downloads" in sc.FOLDERS


def test_open_app_missing_absolute_path_is_friendly(monkeypatch):
    sys = sc.SystemCommands()
    # An absolute path that does not exist must give the friendly
    # "not installed" reply, never a raw Popen error.
    msg = sys.open_app(r"C:\\No\\Such\\Dir\\app.exe", "app")
    assert "couldn't find" in msg.lower()


# ── Circuit breaker ───────────────────────────────────────────

def test_breaker_opens_after_threshold():
    b = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    assert b.is_open is False
    b.record_failure()
    b.record_failure()
    assert b.is_open is False
    b.record_failure()
    assert b.is_open is True


def test_breaker_success_resets():
    b = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.is_open is False  # success reset the streak


def test_breaker_half_open_after_recovery():
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    b.record_failure()
    assert b.is_open is True
    time.sleep(0.02)
    assert b.is_open is False  # half-open: one probe allowed


def test_breaker_disabled_never_opens():
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=60, enabled=False)
    b.record_failure()
    b.record_failure()
    assert b.is_open is False


def test_breaker_decorator_raises_circuit_open():
    b = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    calls = []

    @b
    def risky():
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        risky()
    with pytest.raises(CircuitOpenError):
        risky()  # fast-failed, risky() never ran again
    assert len(calls) == 1


def test_breaker_decorator_records_success():
    b = CircuitBreaker(failure_threshold=2, recovery_timeout=60)

    @b
    def ok():
        return 42

    assert ok() == 42
    b.record_failure()  # one failure, then success resets
    assert ok() == 42
    assert b.is_open is False


# ── Ollama breaker wiring ─────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, lines=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self._lines = lines or []
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


def _ollama_client(monkeypatch, get_response, post_response):
    monkeypatch.setattr(oc.requests, "get", lambda *a, **k: get_response)
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: post_response)
    return OllamaClient(base_url="http://fake:11434", model="qwen3:8b")


def test_ollama_fast_fails_when_breaker_open(monkeypatch):
    """After 3 consecutive failures, further requests never touch the
    network — they fast-fail with the offline behavior."""
    def boom(*a, **k):
        raise oc.requests.ConnectionError("down")

    client = _ollama_client(
        monkeypatch,
        get_response=_FakeResponse(200, {"models": []}),
        post_response=None,
    )
    monkeypatch.setattr(oc.requests, "post", boom)

    # Three failures trip the breaker.
    assert client.ask("hello") is None
    assert client.ask("hello") is None
    assert client.ask("hello") is None
    assert client._breaker.is_open is True

    # Next request fast-fails without hitting the network.
    calls = []
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: calls.append(1))
    assert client.ask("hello") is None
    assert client.ask_stream("hello") is None
    assert calls == []
    assert client.is_available() is False


def test_ollama_breaker_recovers_on_success(monkeypatch):
    def flaky(*a, **k):
        if flaky.n < 3:
            flaky.n += 1
            raise oc.requests.ConnectionError("down")
        return _FakeResponse(200, {"message": {"content": "back online"}})

    flaky.n = 0
    client = _ollama_client(
        monkeypatch,
        get_response=_FakeResponse(200, {"models": []}),
        post_response=None,
    )
    monkeypatch.setattr(oc.requests, "post", flaky)

    # Speed up the recovery window so the half-open probe happens fast.
    client._breaker.recovery_timeout = 0.01

    assert client.ask("hi") is None
    assert client.ask("hi") is None
    assert client.ask("hi") is None
    assert client._breaker.is_open is True
    time.sleep(0.02)
    # Half-open probe succeeds -> breaker closes and the answer flows.
    assert client.ask("hi") == "back online"
    assert client._breaker.is_open is False


def test_ollama_timeout_raises_typed_error(monkeypatch):
    from brain.exceptions import OllamaTimeoutError

    def timeout(*a, **k):
        raise oc.requests.Timeout("slow")

    client = _ollama_client(
        monkeypatch,
        get_response=_FakeResponse(200, {"models": []}),
        post_response=None,
    )
    monkeypatch.setattr(oc.requests, "post", timeout)
    with pytest.raises(OllamaTimeoutError):
        client.ask("hello")


# ── Benchmark baseline ────────────────────────────────────────

def _summary(model, total):
    return {
        "model": model,
        "total": total,
        "ttft": total * 0.4,
        "tokens_per_sec": 10.0,
        "quality": 0.9,
    }


def test_benchmark_save_and_load_baseline(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline([_summary("qwen3:8b", 14.3)], path)
    from jarvis_cli.benchmark import load_baseline

    loaded = load_baseline(path)
    assert loaded["qwen3:8b"]["total"] == 14.3


def test_benchmark_compare_detects_regression(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline([_summary("qwen3:8b", 10.0)], path)
    # 25s vs 10s baseline = 2.5x -> regression.
    regressions = compare_baseline(
        [_summary("qwen3:8b", 25.0)], path
    )
    assert len(regressions) == 1
    assert regressions[0]["model"] == "qwen3:8b"
    assert regressions[0]["ratio"] > REGRESSION_RATIO
    # A faster run is not a regression.
    assert compare_baseline([_summary("qwen3:8b", 8.0)], path) == []


def test_benchmark_compare_missing_baseline_returns_empty(tmp_path):
    assert compare_baseline(
        [_summary("qwen3:8b", 25.0)], tmp_path / "missing.json"
    ) == []
