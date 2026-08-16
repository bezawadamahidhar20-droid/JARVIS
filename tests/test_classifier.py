"""Question classifier tests — the simplified web-search vs local-LLM decision.

classify() returns the plain strings "web_search" / "local_llm";
needs_search() is the boolean equivalent. Time/date questions are now
routed to the command registry, so the classifier never emits tool
intents.
"""

from brain.classifier import QuestionClassifier

WEB_SEARCH = "web_search"
LOCAL_LLM = "local_llm"


def _classifier():
    return QuestionClassifier()


def test_required_acceptance_cases():
    """The eight cases from the requirements must route exactly as specified."""
    c = _classifier()
    assert c.classify("who is the current chief minister of andhra pradesh") == WEB_SEARCH
    assert c.classify("what is python") == LOCAL_LLM
    assert c.classify("what time is it") == LOCAL_LLM
    assert c.classify("what is today's date") == WEB_SEARCH
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


def test_spec_static_questions():
    """Static knowledge questions stay off the web."""
    c = _classifier()
    assert c.classify("what is a binary search tree") == LOCAL_LLM
    assert c.classify("what is recursion") == LOCAL_LLM
    assert c.classify("explain a linked list") == LOCAL_LLM


def test_current_information_triggers_web_search():
    c = _classifier()
    for q in (
        "what is today's weather",
        "what is the latest news",
        "who won the match today",
        "what is the current price of bitcoin",
        "what is the current president of the united states",
        "latest openai news",
        "latest nvidia news",
        "what happened recently",
        "what is the latest version of python",
    ):
        assert c.classify(q) == WEB_SEARCH, f"expected WEB_SEARCH for {q!r}"


def test_office_holder_triggers_web_search():
    """Office holders change over time even without the word 'current'."""
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


def test_empty_input_is_local():
    c = _classifier()
    assert c.classify("") == LOCAL_LLM
    assert c.classify("   ") == LOCAL_LLM


def test_needs_search_boolean_equivalent():
    c = _classifier()
    assert c.needs_search("what is the latest news") is True
    assert c.needs_search("explain recursion") is False
    assert c.needs_search("") is False
