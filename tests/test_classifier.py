"""Question classifier tests — the exact routing cases from the spec."""

from brain.classifier import (
    DATE_TOOL,
    LOCAL_LLM,
    TIME_TOOL,
    WEB_SEARCH,
    QuestionClassifier,
)


def _classifier(mode="auto"):
    return QuestionClassifier(mode=mode)


def test_required_acceptance_cases():
    """The eight cases from the requirements must route exactly as specified."""
    c = _classifier()
    assert c.classify("who is the current chief minister of andhra pradesh") == WEB_SEARCH
    assert c.classify("what is python") == LOCAL_LLM
    assert c.classify("what time is it") == TIME_TOOL
    assert c.classify("what is today's date") == DATE_TOOL
    assert c.classify("what happened today") == WEB_SEARCH
    assert c.classify("who is the current prime minister of india") == WEB_SEARCH
    assert c.classify("explain binary search") == LOCAL_LLM
    assert c.classify("what is the latest python version") == WEB_SEARCH


def test_spec_current_questions():
    """The remaining current-information questions from the spec."""
    c = _classifier()
    for q in (
        "who is the current president of india",
        "what is the latest news in andhra pradesh",
        "what is the current weather in chennai",
    ):
        assert c.classify(q) == WEB_SEARCH, f"expected WEB_SEARCH for {q!r}"


def test_spec_static_and_clock_questions():
    """Static + clock questions from the spec stay off the web."""
    c = _classifier()
    assert c.classify("what is a binary search tree") == LOCAL_LLM
    assert c.classify("what is recursion") == LOCAL_LLM
    assert c.classify("explain a linked list") == LOCAL_LLM
    assert c.classify("what day is today") == DATE_TOOL


def test_current_information_triggers_web_search():
    c = _classifier()
    for q in (
        "what is today's weather",
        "what is the latest news",
        "who won the match today",
        "what is the current price of bitcoin",
        "what is the current president of the united states",
        "what is happening in andhra pradesh",
        "latest openai news",
        "latest nvidia news",
        "what happened recently",
        "what is the latest version of python",
    ):
        assert c.classify(q) == WEB_SEARCH, f"expected WEB_SEARCH for {q!r}"


def test_office_holder_without_strong_marker():
    """'who is the prime minister of india' must still search — office
    holders change over time even without the word 'current'."""
    c = _classifier()
    assert c.classify("who is the prime minister of india") == WEB_SEARCH
    assert c.classify("who is the ceo of microsoft") == WEB_SEARCH


def test_static_knowledge_stays_local():
    c = _classifier()
    for q in (
        "what is python",
        "explain recursion",
        "what is a linked list",
        "explain machine learning",
        "what is an operating system",
        "what is a binary tree",
        "what is the capital of france",
        "explain how gravity works",
        "tell me a joke",
        "how does bitcoin mining work",
    ):
        assert c.classify(q) == LOCAL_LLM, f"expected LOCAL_LLM for {q!r}"


def test_historical_questions_stay_local():
    c = _classifier()
    for q in (
        "who was the first president of india",
        "who was the prime minister during world war two",
        "who won the 1994 world cup",
        "what was the population of india in 1950",
        "who was the ceo of microsoft in 2000",
    ):
        assert c.classify(q) == LOCAL_LLM, f"expected LOCAL_LLM for {q!r}"


def test_greetings_with_strong_markers_stay_local():
    c = _classifier()
    assert c.classify("how are you today") == LOCAL_LLM
    assert c.classify("whats up") == LOCAL_LLM


def test_electric_current_not_time_sensitive():
    c = _classifier()
    assert c.classify("what is electric current") == LOCAL_LLM


def test_local_mode_never_searches():
    c = _classifier(mode="local")
    assert c.classify("what happened today") == LOCAL_LLM
    assert c.classify("who is the current chief minister") == LOCAL_LLM
    # Time/date still deterministic in local mode.
    assert c.classify("what time is it") == TIME_TOOL
    assert c.classify("what is today's date") == DATE_TOOL


def test_web_mode_always_searches():
    c = _classifier(mode="web")
    assert c.classify("what is python") == WEB_SEARCH
    assert c.classify("explain recursion") == WEB_SEARCH
    # Time/date still deterministic in web mode.
    assert c.classify("what time is it") == TIME_TOOL


def test_empty_input_is_local():
    c = _classifier()
    assert c.classify("") == LOCAL_LLM
    assert c.classify("   ") == LOCAL_LLM


def test_unknown_mode_falls_back_to_auto():
    c = _classifier(mode="banana")
    assert c.mode == "auto"
