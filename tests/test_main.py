"""JARVIS orchestrator tests — all subsystems faked, no hardware/AI calls."""


from brain.memory import ConversationMemory
from brain.router import IntentRouter
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

    def __init__(self, utterances=None):
        # A queue of utterances to return from listen(); empty = silent.
        self.utterances = list(utterances or [])

    def load(self):
        return True

    def listen(self):
        if self.utterances:
            return self.utterances.pop(0)
        return None

    def unload(self):
        pass


class FakeMic:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def describe(self):
        return "fake mic"


class FakeProvider:
    name = "fake"

    def __init__(self, available=True, reply="A concise fake answer."):
        self.available = available
        self.reply = reply
        self.asked = []

    def is_available(self):
        return self.available

    def ask(self, user_input, memory=None, context=None):
        self.asked.append(user_input)
        return self.reply

    def ask_stream(self, user_input, memory=None, on_sentence=None, context=None):
        self.asked.append(user_input)
        if on_sentence:
            on_sentence(self.reply)
        return self.reply

    def describe(self):
        return "fake provider"


def make_jarvis(provider=None, text_mode=True, stt=None):
    return JARVIS(
        text_mode=text_mode,
        components={
            "mic": FakeMic(),
            "stt": stt if stt is not None else FakeSTT(),
            "tts": FakeTTS(),
            "memory": ConversationMemory(max_turns=6, max_chars=3000, persist_path=""),
            "provider": provider if provider is not None else FakeProvider(),
            "router": IntentRouter(),
            "commands": CommandRegistry(),
        },
    )


def test_exit_stops_jarvis():
    jarvis = make_jarvis()
    assert jarvis.process_input("goodbye") is None
    assert jarvis.running is False


def test_exit_variants():
    jarvis = make_jarvis()
    for phrase in ("exit", "quit", "shutdown jarvis", "bye"):
        jarvis.running = True
        assert jarvis.process_input(phrase) is None, phrase
        assert jarvis.running is False, phrase


def test_fast_response_returns_truthy():
    jarvis = make_jarvis()
    assert jarvis.process_input("hello")
    assert jarvis.tts.spoken  # something was spoken


def test_command_returns_truthy():
    jarvis = make_jarvis()
    assert jarvis.process_input("what time is it")
    assert any("It's" in s for s in jarvis.tts.spoken)


def test_ai_question_uses_provider():
    provider = FakeProvider(reply="Python is a language.")
    jarvis = make_jarvis(provider=provider)
    assert jarvis.process_input("what is python")
    assert provider.asked == ["what is python"]
    # Response spoken + stored in memory.
    assert "Python is a language." in jarvis.tts.spoken
    assert len(jarvis.memory) == 2  # user + assistant


def test_ai_stream_uses_sentences_in_voice_mode():
    provider = FakeProvider(reply="First sentence. Second sentence.")
    jarvis = make_jarvis(provider=provider, text_mode=False)
    assert jarvis.process_input("explain recursion")
    assert provider.asked == ["explain recursion"]


def test_normal_questions_keep_jarvis_listening():
    """A normal question must NEVER terminate JARVIS: the loop answers
    it and keeps listening until an explicit exit phrase."""
    import threading

    # "explain ..." queries route to the AI provider (the classifier
    # sends "what is X" to web search, which needs a real API key).
    provider = FakeProvider(reply="Data size is the amount of information.")
    jarvis = make_jarvis(
        provider=provider,
        text_mode=False,
        stt=FakeSTT(utterances=["explain data size", "explain recursion", "goodbye"]),
    )
    jarvis.running = True  # run() sets this; the test drives the loop directly
    thread = threading.Thread(
        target=jarvis.start_listening_loop, daemon=True, name="jarvis-test-loop"
    )
    thread.start()
    thread.join(timeout=30.0)
    assert not thread.is_alive()

    # BOTH questions were answered before the explicit goodbye.
    assert provider.asked == ["explain data size", "explain recursion"]
    assert "Data size is the amount of information." in jarvis.tts.spoken
    # ...and the goodbye produced the farewell.
    assert any("Goodbye" in s for s in jarvis.tts.spoken)
    jarvis.shutdown()


def test_shutdown_cancels_running_loop(capsys):
    """shutdown() must cancel a pending listening loop cleanly — prompt
    return, no 'Task was destroyed but it is pending!' warning."""
    import threading
    import time

    jarvis = make_jarvis(
        text_mode=False,
        stt=FakeSTT(utterances=[]),  # silent mic: loop stays in LISTENING
    )
    jarvis.running = True  # run() sets this; the test drives the loop directly
    thread = threading.Thread(
        target=jarvis.start_listening_loop, daemon=True, name="jarvis-test-loop"
    )
    thread.start()
    # Wait until the loop task is actually registered (poll, so the
    # test is not sensitive to machine load).
    deadline = time.monotonic() + 10.0
    while jarvis._main_task is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jarvis._main_task is not None, "loop task never registered"
    jarvis.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert "JARVIS offline" in capsys.readouterr().out


def test_shutdown_races_loop_startup_does_not_hang():
    """shutdown() called immediately after starting the loop must not
    hang the loop thread — the future is recorded before blocking, so
    cancellation always unblocks it even before the task starts."""
    import threading

    jarvis = make_jarvis(text_mode=False, stt=FakeSTT(utterances=[]))
    jarvis.running = True
    thread = threading.Thread(
        target=jarvis.start_listening_loop, daemon=True, name="jarvis-test-loop"
    )
    thread.start()
    # Shut down immediately — the loop task may not have started yet.
    jarvis.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_provider_offline_speaks_error_and_continues():
    provider = FakeProvider(available=False)
    jarvis = make_jarvis(provider=provider)
    assert jarvis.ollama_ok is False
    # Local commands still work.
    assert jarvis.process_input("what time is it")
    # AI questions get the offline message, not a crash.
    assert jarvis.process_input("what is python")
    assert any("offline" in s.lower() for s in jarvis.tts.spoken)


def test_empty_input_ignored():
    jarvis = make_jarvis()
    assert jarvis.process_input("") is None
    assert jarvis.process_input("   ") is None


def test_clear_memory():
    jarvis = make_jarvis()
    jarvis.process_input("what is python")
    assert len(jarvis.memory) == 2
    jarvis.process_input("clear memory")
    assert len(jarvis.memory) == 0


def test_shutdown_releases_resources(capsys):
    jarvis = make_jarvis()
    jarvis.shutdown()
    out = capsys.readouterr().out
    assert "JARVIS offline" in out


def test_benchmark_report(monkeypatch, capsys):
    provider = FakeProvider()
    jarvis = make_jarvis(provider=provider)
    jarvis.benchmark = True
    jarvis._record_turn({"listen": 0.5, "process": 1.2, "speak": 0.3})
    jarvis._record_turn({"listen": 0.4, "process": 0.9, "speak": 0.2})
    jarvis._print_benchmark()
    out = capsys.readouterr().out
    assert "JARVIS BENCHMARK" in out
    assert "Turns: 2" in out


def test_initialize_sequence_runs_with_fakes(capsys):
    jarvis = make_jarvis()
    jarvis.initialize()
    out = capsys.readouterr().out
    assert "Configuration loaded" in out
    assert "Command router ready" in out
    assert jarvis.router is not None
    assert jarvis.commands is not None
