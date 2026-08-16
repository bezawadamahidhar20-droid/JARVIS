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

    def load(self):
        return True

    def listen(self):
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

    def ask(self, user_input, memory=None):
        self.asked.append(user_input)
        return self.reply

    def ask_stream(self, user_input, memory=None, on_sentence=None):
        self.asked.append(user_input)
        if on_sentence:
            on_sentence(self.reply)
        return self.reply

    def describe(self):
        return "fake provider"


def make_jarvis(provider=None, text_mode=True):
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
        },
    )


def test_exit_returns_false():
    jarvis = make_jarvis()
    assert jarvis.process_input("goodbye") is False


def test_exit_variants():
    jarvis = make_jarvis()
    for phrase in ("exit", "quit", "shutdown jarvis", "bye"):
        assert jarvis.process_input(phrase) is False, phrase


def test_fast_response_returns_true():
    jarvis = make_jarvis()
    assert jarvis.process_input("hello") is True
    assert jarvis.tts.spoken  # something was spoken


def test_command_returns_true():
    jarvis = make_jarvis()
    assert jarvis.process_input("what time is it") is True
    assert any("It's" in s for s in jarvis.tts.spoken)


def test_ai_question_uses_provider():
    provider = FakeProvider(reply="Python is a language.")
    jarvis = make_jarvis(provider=provider)
    assert jarvis.process_input("what is python") is True
    assert provider.asked == ["what is python"]
    # Response spoken + stored in memory.
    assert "Python is a language." in jarvis.tts.spoken
    assert len(jarvis.memory) == 2  # user + assistant


def test_ai_stream_uses_sentences_in_voice_mode():
    provider = FakeProvider(reply="First sentence. Second sentence.")
    jarvis = make_jarvis(provider=provider, text_mode=False)
    assert jarvis.process_input("explain recursion") is True
    assert provider.asked == ["explain recursion"]


def test_provider_offline_speaks_error_and_continues():
    provider = FakeProvider(available=False)
    jarvis = make_jarvis(provider=provider)
    assert jarvis.ollama_ok is False
    # Local commands still work.
    assert jarvis.process_input("what time is it") is True
    # AI questions get the offline message, not a crash.
    assert jarvis.process_input("what is python") is True
    assert any("offline" in s.lower() for s in jarvis.tts.spoken)


def test_empty_input_ignored():
    jarvis = make_jarvis()
    assert jarvis.process_input("") is True
    assert jarvis.process_input("   ") is True


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
