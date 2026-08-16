"""Issue 3 — improved LLM sentence splitting.

Complete sentences (``.`` ``!`` ``?``, newlines) must be emitted as
soon as they finish so TTS can start; common abbreviations must NOT
split mid-abbreviation.
"""

from brain.text_utils import clean_response, split_into_sentences
from brain.ollama_client import OllamaClient


def _split(buffer):
    """Convenience: sentences only (remainder discarded)."""
    sentences, _ = split_into_sentences(buffer)
    return sentences


# ── Shared splitter (brain/text_utils.py) ─────────────────────

def test_two_sentences():
    assert _split("Hello. How are you?") == ["Hello.", "How are you?"]


def test_exclamation_and_question():
    assert _split("Open Chrome! Then search Google?") == [
        "Open Chrome!",
        "Then search Google?",
    ]


def test_newline_split():
    assert _split("Line one\nLine two.") == ["Line one", "Line two."]
    assert _split("First.\nSecond!") == ["First.", "Second!"]


def test_abbreviation_does_not_split():
    assert _split("Dr. Smith is here. He is a doctor.") == [
        "Dr. Smith is here.",
        "He is a doctor.",
    ]
    assert _split("It is 3 p.m. and all is well.") == [
        "It is 3 p.m. and all is well.",
    ]


def test_single_initial_does_not_split():
    assert _split("J. Smith wrote it. It is good.") == [
        "J. Smith wrote it.",
        "It is good.",
    ]


def test_ellipsis_does_not_split_into_junk():
    """The three dots form one speech chunk — never 'Hmm.', '.', '.'."""
    sentences, _ = split_into_sentences("Hmm... okay.")
    assert sentences == ["Hmm...", "okay."]


def test_punctuation_only_skipped():
    assert _split("...") == []
    assert _split("!?") == []


def test_partial_remainder_kept_for_streaming():
    sentences, remainder = split_into_sentences("Hello. How are")
    assert sentences == ["Hello."]
    assert remainder == " How are"


def test_streaming_incremental_chunks():
    """Chunks arrive incrementally; each finished sentence is emitted
    immediately instead of waiting for the whole response."""
    buffer = ""
    emitted = []
    for chunk in ("Hello", " there.", " How ", "are you?"):
        buffer += chunk
        sentences, buffer = split_into_sentences(buffer)
        emitted.extend(sentences)
    assert emitted == ["Hello there.", "How are you?"]
    assert buffer == ""


def test_clean_response_strips_markdown():
    assert clean_response("**bold** and `code`") == "bold and code"
    assert clean_response("") == ""


# ── Ollama client integration ─────────────────────────────────

def test_ollama_extract_sentences_delegates():
    client = OllamaClient(base_url="http://fake", model="m")
    sentences, remainder = client._extract_sentences(
        "Hello. How are you?"
    )
    assert sentences == ["Hello.", "How are you?"]
    assert remainder == ""


def test_ollama_extract_sentences_abbreviations():
    client = OllamaClient(base_url="http://fake", model="m")
    sentences, _ = client._extract_sentences(
        "Dr. Smith is here. Open Chrome! Then search Google?"
    )
    assert sentences == [
        "Dr. Smith is here.",
        "Open Chrome!",
        "Then search Google?",
    ]


def test_ollama_stream_emits_per_sentence(monkeypatch):
    """Streaming must emit the first sentence before the stream ends."""
    import brain.ollama_client as oc

    lines = [
        '{"message":{"content":"Open Chrome! "},"done":false}',
        '{"message":{"content":"Then search Google?"},"done":false}',
        '{"message":{"content":""},"done":true}',
    ]

    class FakeResp:
        status_code = 200

        def iter_lines(self, decode_unicode=False):
            return iter(lines)

        def close(self):
            pass

    monkeypatch.setattr(oc.requests, "get",
                        lambda *a, **k: type("R", (), {"status_code": 200, "json": lambda s: {"models": []}})())
    monkeypatch.setattr(oc.requests, "post", lambda *a, **k: FakeResp())
    client = OllamaClient(base_url="http://fake:11434", model="qwen3:8b")

    spoken = []
    result = client.ask_stream("hi", memory=None, on_sentence=spoken.append)
    assert result == "Open Chrome! Then search Google?"
    assert spoken == ["Open Chrome!", "Then search Google?"]
