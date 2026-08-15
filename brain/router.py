"""Intent router — decides whether an utterance is a command or a question.

CRITICAL RULE
-------------
Everything that is NOT explicitly recognised as a command falls through to
``AI_QUESTION``. This is the single fix that makes general questions work:
"What is Python?", "Who are you?", "Tell me a joke" etc. are all routed to
Qwen3 instead of hitting a dead "I don't understand" branch.

Intent types
------------
* ``COMMAND``      — local action (open app/website, time/date, screenshot)
* ``AI_QUESTION``  — send to the LLM (the default)
* ``EXIT``         — goodbye, shut down
* ``CLEAR_MEMORY`` — forget the conversation
"""

import re

from commands.time_commands import DATE_RE, TIME_RE

# ── Intent constants ──────────────────────────────────────────────────────────

COMMAND = "COMMAND"
AI_QUESTION = "AI_QUESTION"
EXIT = "EXIT"
CLEAR_MEMORY = "CLEAR_MEMORY"

# ── Patterns (pre-compiled once for speed) ────────────────────────────────────

# A leading question word turns "how do I open chrome?" into a question for
# the AI instead of a command — the user is ASKING, not ordering.
_QUESTION_WORD_RE = re.compile(
    r"^(how|what|why|when|where|which|who|whom|whose|can you|could you|"
    r"do you|are you|is it|should i|tell me how|how do i)\b",
    re.IGNORECASE,
)

# exit/quit/goodbye as the ENTIRE utterance (avoid "what is an exit code?").
# "Goodbye JARVIS" / "Goodbye sir" are also accepted since users say them
# naturally; the key guard is that "exit"/"quit"/"stop" alone only fire when
# they are the whole utterance, never inside a real question.
_EXIT_RE = re.compile(
    r"^\s*(?:exit|quit|goodbye|good bye|bye|stop)\s*$"
    r"|^\s*(?:goodbye|good bye|bye)\s+(?:jarvis|sir)?\s*$"
    r"|^\s*stop\s+jarvis\s*$",
    re.IGNORECASE,
)

_CLEAR_RE = re.compile(r"\b(clear|reset|forget)\s+(the\s+)?(memory|history|conversation)\b", re.IGNORECASE)

_SCREENSHOT_RE = re.compile(r"\b(take\s+(a\s+)?screenshot|screenshot)\b", re.IGNORECASE)

# open/launch/start/go to/visit <something>
_OPEN_RE = re.compile(
    r"\b(?:open|launch|start|go\s+to|visit)\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)


class IntentRouter:
    """Maps raw text to an ``(intent, text)`` tuple."""

    def __init__(self) -> None:
        # Keep references to the compiled patterns so they are compiled
        # exactly once and re-used on every route() call.
        self._question_word = _QUESTION_WORD_RE
        self._exit = _EXIT_RE
        self._clear = _CLEAR_RE
        self._screenshot = _SCREENSHOT_RE
        self._open = _OPEN_RE
        self._time = TIME_RE
        self._date = DATE_RE

    def route(self, text: str) -> tuple[str, str]:
        """Return ``(intent, text)`` for the given utterance."""
        t = (text or "").strip()
        if not t:
            return AI_QUESTION, t

        # 1. Explicit "goodbye" — full-utterance match only.
        if self._exit.match(t):
            return EXIT, t

        # 2. Conversation hygiene commands.
        if self._clear.search(t):
            return CLEAR_MEMORY, t

        # 3. Clock / calendar (specific phrases, checked BEFORE the generic
        #    question-word guard so "what time is it?" still becomes COMMAND).
        if self._time.search(t) or self._date.search(t):
            return COMMAND, t

        # 4. Screenshots.
        if self._screenshot.search(t):
            return COMMAND, t

        # 5. open/launch/go to <target>. Guarded so that sentences that merely
        #    START like a question ("how do I open chrome?") go to the AI.
        if not self._question_word.match(t) and self._open.search(t):
            return COMMAND, t

        # 6. DEFAULT: everything else is a question for the AI.
        return AI_QUESTION, t