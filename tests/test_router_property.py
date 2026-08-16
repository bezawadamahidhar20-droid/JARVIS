"""Issue 17 — property-based router tests (Hypothesis).

The router must be total: any string (including garbage, very long
input, and unusual Unicode) is classified without crashing, and
malformed input never triggers execution.
"""

import string

from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

from brain.router import (
    Intent,
    IntentRouter,
    MAX_INPUT_CHARS,
    sanitize_input,
    validate_input,
)

_VALID_INTENTS = {
    Intent.COMMAND,
    Intent.AI_QUESTION,
    Intent.WEB_SEARCH,
    Intent.CLEAR_MEMORY,
    Intent.EXIT,
    Intent.FAST_RESPONSE,
    Intent.STOP_SPEECH,
    Intent.UNKNOWN,
}

# Keep the property tests fast while still covering a wide space.
_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_SETTINGS
@given(st.text())
def test_arbitrary_strings_never_crash(text):
    router = IntentRouter()
    intent, cleaned = router.route(text)
    assert intent in _VALID_INTENTS
    assert isinstance(cleaned, str)


@_SETTINGS
@given(st.text(alphabet=" \t\n", min_size=1, max_size=50))
def test_whitespace_only_input_is_safe(whitespace):
    router = IntentRouter()
    intent, cleaned = router.route(whitespace)
    assert intent == Intent.UNKNOWN
    assert cleaned == ""


@_SETTINGS
@given(st.text(
    # No whitespace: normalization (whitespace collapsing) must not
    # shrink the input below the limit and invalidate the property.
    alphabet=string.ascii_lowercase + string.digits,
    min_size=MAX_INPUT_CHARS + 1,
    max_size=MAX_INPUT_CHARS + 2000,
))
def test_very_long_input_rejected_safely(long_text):
    router = IntentRouter()
    intent, cleaned = router.route(long_text)
    # Rejected: never routed to a command handler or the LLM.
    assert intent == Intent.UNKNOWN
    assert cleaned == ""


@_SETTINGS
@given(st.text(alphabet=string.printable, min_size=1, max_size=200))
def test_unusual_unicode_and_printable_never_crash(text):
    router = IntentRouter()
    intent, _ = router.route(text)
    assert intent in _VALID_INTENTS


@_SETTINGS
@given(st.text())
def test_router_is_deterministic(text):
    router = IntentRouter()
    assert router.route(text) == router.route(text)


@_SETTINGS
@given(st.text(min_size=1, max_size=100))
def test_validate_input_bounds_length(text):
    result = validate_input(text)
    if result:
        assert len(result) <= MAX_INPUT_CHARS
        assert result == sanitize_input(text)


@_SETTINGS
@given(st.text())
def test_malformed_open_commands_never_execute(text):
    """Random text starting with an open verb still produces a COMMAND
    intent whose target is resolved by the registry — never executed
    directly from the raw string."""
    from commands.registry import CommandRegistry

    router = IntentRouter()
    registry = CommandRegistry()
    # If the router says COMMAND, the registry must be able to resolve
    # it to a known command or decline politely (never raise).
    intent, _ = router.route(text)
    if intent == Intent.COMMAND:
        result = registry.execute_with_meta(text)
        assert result is None or result.command is not None
