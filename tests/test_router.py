"""Intent router tests — deterministic classification."""

from brain.router import IntentRouter, Intent, parse_model_mode


def test_exit_phrases():
    router = IntentRouter()
    for phrase in ("goodbye", "exit", "quit", "shutdown jarvis",
                   "bye", "good bye"):
        intent, _ = router.route(phrase)
        assert intent == Intent.EXIT, f"expected EXIT for {phrase!r}"


def test_memory_clear_phrases():
    router = IntentRouter()
    for phrase in ("clear memory", "reset the conversation",
                   "forget everything", "wipe history"):
        intent, _ = router.route(phrase)
        assert intent == Intent.CLEAR_MEMORY, f"expected CLEAR for {phrase!r}"


def test_fast_responses():
    router = IntentRouter()
    for phrase in ("hello", "hi jarvis", "who are you",
                   "good morning", "thank you"):
        intent, reply = router.route(phrase)
        assert intent == Intent.FAST_RESPONSE
        assert reply


def test_command_phrases_stay_ai_question():
    """The router no longer emits a COMMAND intent — commands are
    resolved by the command registry after routing (main.py falls back
    to the registry for any AI_QUESTION)."""
    router = IntentRouter()
    for phrase in ("what time is it", "what's the date",
                   "open chrome", "open notepad", "open calculator",
                   "open youtube", "take a screenshot",
                   "tell me the time", "what day is it today"):
        intent, _ = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"


def test_unknown_goes_to_ai():
    router = IntentRouter()
    for phrase in ("what is python", "explain recursion",
                   "tell me a joke", "how does gravity work"):
        intent, _ = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"


def test_empty_input_unknown():
    router = IntentRouter()
    intent, cleaned = router.route("")
    assert intent == Intent.UNKNOWN
    assert cleaned == ""


def test_case_insensitive():
    router = IntentRouter()
    intent, _ = router.route("WHAT TIME IS IT")
    assert intent == Intent.AI_QUESTION
    intent, _ = router.route("Open Chrome")
    assert intent == Intent.AI_QUESTION


def test_time_date_variants_are_not_tools():
    """Time/date questions are plain AI_QUESTION routes (the command
    registry handles them downstream), never the LLM classifier."""
    router = IntentRouter()
    for phrase in (
        "what is today date",      # Whisper drops the apostrophe
        "what day is today",
        "whats the time now",
        "what time is it now",
        "tell me the date",
        "what is the current time",
    ):
        intent, _ = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"


def test_web_search_questions_route_to_ai():
    """Current-information questions route as AI_QUESTION; the
    classifier in main() decides whether to consult the web."""
    router = IntentRouter()
    for phrase in (
        "who is the current chief minister of andhra pradesh",
        "what is the latest news",
        "what happened today",
        "what is today's weather",
    ):
        intent, _ = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"


def test_stop_speaking_intent():
    router = IntentRouter()
    for phrase in ("stop speaking", "stop talking", "be quiet"):
        intent, _ = router.route(phrase)
        assert intent == Intent.STOP_SPEECH, f"expected STOP for {phrase!r}"


# ── Runtime model-mode control ────────────────────────────────

def test_model_mode_switch_phrases():
    router = IntentRouter()
    for phrase, expected in (
        ("switch to fast mode", ("switch", "fast")),
        ("use the fast model", ("switch", "fast")),
        ("use the fast one", ("switch", "fast")),
        ("go fast mode", ("switch", "fast")),
        ("switch to quality mode", ("switch", "quality")),
        ("use the quality model", ("switch", "quality")),
        ("change to quality mode", ("switch", "quality")),
        ("go into fast mode", ("switch", "fast")),
    ):
        intent, cleaned = router.route(phrase)
        assert intent == Intent.MODEL_MODE, f"expected MODEL_MODE for {phrase!r}"
        assert cleaned == expected, f"wrong mode for {phrase!r}"


def test_model_mode_status_phrases():
    router = IntentRouter()
    for phrase in (
        "which model are you using",
        "what model are you running",
        "what model are you on",
        "what model do you use",
        "what mode are you in",
    ):
        intent, cleaned = router.route(phrase)
        assert intent == Intent.MODEL_MODE, f"expected MODEL_MODE for {phrase!r}"
        assert cleaned == ("status", "")


def test_non_model_mode_phrases_stay_ai():
    """Unknown modes and ordinary questions must NOT be swallowed."""
    router = IntentRouter()
    for phrase in (
        "switch to turbo mode",
        "use the banana model",
        "explain fast mode",
        "what is fast mode",
        "which model is better for gaming",
    ):
        intent, _ = router.route(phrase)
        assert intent == Intent.AI_QUESTION, f"expected AI for {phrase!r}"


def test_parse_model_mode():
    assert parse_model_mode("switch to fast mode") == ("switch", "fast")
    assert parse_model_mode("use the quality model") == ("switch", "quality")
    assert parse_model_mode("which model are you using") == ("status", "")
    assert parse_model_mode("what is python") is None
    assert parse_model_mode("") is None
