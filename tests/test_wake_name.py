"""Wake-name normalization tests.

Whisper sometimes hears "JARVIS" as a near-homophone ("lajad", "jervis",
...). We normalize those variants to "jarvis" only in *address position*
(start of the utterance or after a pause word / comma). Ordinary user
sentences must never be altered.
"""

from brain.router import Intent, IntentRouter, normalize_wake_name


def test_address_position_variants_normalized():
    assert normalize_wake_name("so lajad can you listen to me") == (
        "so jarvis can you listen to me"
    )
    assert normalize_wake_name("hi jervis") == "hi jarvis"
    assert normalize_wake_name("lajav open notepad") == "jarvis open notepad"
    assert normalize_wake_name("hey jarivs what time is it") == (
        "hey jarvis what time is it"
    )
    assert normalize_wake_name("so, lajad, can you listen") == (
        "so, jarvis, can you listen"
    )
    assert normalize_wake_name("please jervis open chrome") == (
        "please jarvis open chrome"
    )


def test_ordinary_sentences_untouched():
    # The variants are common words in normal speech; without address
    # position they must pass through unchanged.
    assert normalize_wake_name("my name is lajad") == "my name is lajad"
    assert normalize_wake_name("tell me about lajad") == "tell me about lajad"
    assert normalize_wake_name("i met jervis yesterday") == "i met jervis yesterday"
    assert normalize_wake_name("the jarivs project is on github") == (
        "the jarivs project is on github"
    )


def test_exact_jarvis_unchanged():
    assert normalize_wake_name("jarvis open notepad") == "jarvis open notepad"


def test_empty_and_whitespace():
    assert normalize_wake_name("") == ""
    assert normalize_wake_name("   ") == "   "


def test_router_fast_response_with_variant():
    router = IntentRouter()
    intent, _ = router.route("hi jervis")
    assert intent == Intent.FAST_RESPONSE


def test_router_command_with_variant():
    router = IntentRouter()
    intent, _ = router.route("lajad open notepad")
    assert intent == Intent.COMMAND
