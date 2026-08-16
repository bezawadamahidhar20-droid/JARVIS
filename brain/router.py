"""
brain/router.py — Intent Router

Decides whether user input is:
  1. A system COMMAND    (open app, time, date, screenshot, status, ...)
  2. A WEB_SEARCH        (current information — needs fresh facts)
  3. An AI QUESTION      (route to Qwen3 via Ollama)
  4. EXIT                (shut down JARVIS)
  5. CLEAR_MEMORY        (reset conversation context)
  6. FAST_RESPONSE       (instant canned reply to a greeting)
  7. STOP_SPEECH         (interrupt TTS output)
  8. MODEL_MODE          (runtime fast/quality model switch + status)

CRITICAL RULE:
Anything that is NOT matched as a command, a web-search question, or a
fast response becomes an AI_QUESTION. This is the fix that makes JARVIS
actually answer normal questions instead of saying
"I don't understand".

Current-information questions are detected by QuestionClassifier
(brain/classifier.py) and routed to WEB_SEARCH instead of the local LLM,
so JARVIS never answers "who is the current ..." from stale training
data (see AI_MODE in .env: auto | local | web).
"""

import re
from typing import Tuple, List, Pattern

from config import jarvis_config
from utils.logger import get_logger

logger = get_logger("router")

__all__ = [
    "Intent",
    "IntentRouter",
    "sanitize_input",
    "validate_input",
    "normalize_wake_name",
    "parse_model_mode_request",
    "FAST_RESPONSES",
]


class Intent:
    """Intent type constants."""
    COMMAND = "command"
    AI_QUESTION = "ai_question"
    WEB_SEARCH = "web_search"
    CLEAR_MEMORY = "clear_memory"
    EXIT = "exit"
    FAST_RESPONSE = "fast_response"
    STOP_SPEECH = "stop_speech"
    MODEL_MODE = "model_mode"
    UNKNOWN = "unknown"


# ── Fast responses ────────────────────────────────────────────
# Greetings answered instantly from this local table instead of
# a full AI round-trip. Keys are normalized (lowercase, no punctuation).
OWNER = jarvis_config.OWNER
ENABLE_FAST_RESPONSES = jarvis_config.ENABLE_FAST_RESPONSES
MAX_INPUT_CHARS = jarvis_config.MAX_INPUT_CHARS


# ── Input sanitisation ────────────────────────────────────────
# Every utterance passes through validate_input() before the router
# (or any command handler) sees it. This guarantees:
#   * empty / whitespace-only input is rejected,
#   * abnormally long input is rejected safely (never reaching
#     command handlers or the LLM),
#   * normal voice commands pass through unchanged (only surrounding
#     whitespace is trimmed and internal whitespace collapsed).


def sanitize_input(text: str | None) -> str:
    """Normalize raw input: strip, collapse internal whitespace runs.

    Returns '' for None/empty input. Does NOT enforce the length cap —
    use validate_input() for the full check.
    """
    if not text:
        return ""
    return " ".join(text.strip().split())


def validate_input(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> str:
    """
    Validate and normalize one user utterance.

    Returns the normalized text, or '' when the input is invalid
    (empty / whitespace-only / exceeds ``max_chars``).
    """
    cleaned = sanitize_input(text)
    if not cleaned:
        return ""
    if max_chars > 0 and len(cleaned) > max_chars:
        logger.warning(
            f"Input rejected: {len(cleaned)} chars exceeds the "
            f"{max_chars} char limit."
        )
        return ""
    return cleaned

_FAST_TEMPLATES: dict = {
    "hello": "Hello, {owner}. How can I help you today?",
    "hello jarvis": "Hello, {owner}. At your service.",
    "hello there": "Hello, {owner}. At your service.",
    "hi": "Hi there, {owner}.",
    "hi jarvis": "Hi, {owner}. How may I assist you?",
    "hey": "Hey, {owner}.",
    "hey jarvis": "Hey, {owner}. Ready when you are.",
    "good morning": "Good morning, {owner}.",
    "good afternoon": "Good afternoon, {owner}.",
    "good evening": "Good evening, {owner}.",
    "good night": "Good night, {owner}. I'll be here if you need me.",
    "thanks": "You're welcome, {owner}.",
    "thank you": "You're welcome, {owner}.",
    "thanks jarvis": "You're welcome, {owner}.",
    "thank you jarvis": "You're welcome, {owner}.",
    "how are you": "I'm running at full capacity, {owner}. How can I help?",
    "how are you today": "I'm running at full capacity, {owner}. How can I help?",
    "how are you doing": "All systems nominal, {owner}.",
    "how are you doing today": "All systems nominal, {owner}.",
    "how is it going": "All systems nominal, {owner}.",
    "hows it going": "All systems nominal, {owner}.",
    "whats up": "Just monitoring the house, {owner}. How can I help?",
    "who are you": "I am JARVIS, your personal AI assistant, {owner}.",
    "who are you jarvis": "I am JARVIS, your personal AI assistant, {owner}.",
    "what is your name": "I am JARVIS, at your service, {owner}.",
    "are you jarvis": "Yes, {owner}. I am JARVIS.",
}


def _normalize_key(text: str) -> str:
    """Normalize a phrase for dictionary lookup."""
    text = re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


FAST_RESPONSES: dict = {
    key: reply.format(owner=OWNER)
    for key, reply in _FAST_TEMPLATES.items()
}


# ── Wake-name normalization ───────────────────────────────────
# Whisper sometimes hears the assistant's name as a near-homophone
# ("jervis", "lajav", "lajad", "jarves", "gervis", ...). We normalize
# those variants to "jarvis" ONLY when they appear in *address position*
# (start of the utterance, or right after a comma / pause word like
# "so"/"hey"). Ordinary sentences are never altered, and the raw
# transcription is kept in the debug log.
_WAKE_VARIANTS = (
    # existing variants
    "jervis", "lajav", "lajad", "jarivs", "jerivs",
    # expanded: common Whisper misrecognitions
    "javas", "jarves", "jarvus", "jarfis", "gervis", "garvis",
    "djarvis", "charvis", "jarvas", "jarvs", "jarvi", "jervas",
    "jarbis", "jarvisa", "jarwis", "jarivis", "jarvish",
    "javris", "jerfis", "jervas", "djarvus", "charlvis",
)
_WAKE_VARIANTS_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _WAKE_VARIANTS) + r")\b",
    re.IGNORECASE,
)
_PAUSE_WORDS = (
    "so", "hey", "hi", "hello", "yo", "okay", "ok", "please",
    "now", "right", "well", "alright",
)


def _is_address_position(prefix: str) -> bool:
    """True when a wake-variant at this position is used as a name."""
    p = prefix.strip()
    if not p:
        return True  # start of the utterance
    if p[-1] in ",;:!?.":
        return True  # after a pause / comma ("so, lajad, ...")
    last_word = re.sub(r"[^a-z']", "", p.split()[-1].lower()) if p.split() else ""
    return last_word in _PAUSE_WORDS


def normalize_wake_name(text: str) -> str:
    """
    Rewrite misrecognized variants of JARVIS's name to "jarvis", but
    only in address position. Returns the text unchanged otherwise.
    """
    if not text or not text.strip():
        return text
    out: list[str] = []
    last = 0
    for m in _WAKE_VARIANTS_RE.finditer(text):
        prefix = text[last:m.start()]
        if _is_address_position(prefix):
            out.append(prefix)
            out.append("jarvis")
            last = m.end()
    out.append(text[last:])
    return "".join(out)


# ── Runtime model-mode control ────────────────────────────────
# "Switch to fast mode" / "which model are you using" are handled
# deterministically — no LLM round-trip, instant reply.

# Switch requests: an explicit verb + a mode word + mode/model.
_MODEL_MODE_SWITCH_RE = re.compile(
    r"\b(switch|change|set|use|go)\s+(to\s+|into\s+)?(the\s+)?"
    r"(fast|quick|speed|quality|accurate|full)\s+(mode|model)\b",
    re.IGNORECASE,
)
# Shorter forms: "use the fast one".
_MODEL_MODE_USE_RE = re.compile(
    r"\buse\s+(the\s+)?(fast|quality)\s+(one|model)\b",
    re.IGNORECASE,
)
# "set/switch the model to fast".
_MODEL_MODE_TO_RE = re.compile(
    r"\b(switch|change|set)\s+(the\s+)?model\s+"
    r"(to\s+|over\s+to\s+)?(fast|quality)\b",
    re.IGNORECASE,
)
# Status requests: ask what model/mode JARVIS is on.
_MODEL_MODE_STATUS_RE = re.compile(
    r"\b(which|what)\s+model\s+(are you (using|running|on)|do you use)\b"
    r"|\bwhat\s+mode\s+(are you (in|using)|is this)\b",
    re.IGNORECASE,
)
# Modes we can switch to (fast words -> fast mode, quality words -> quality).
_FAST_MODE_WORD_RE = re.compile(r"\b(fast|quick|speed)\b", re.IGNORECASE)
_QUALITY_MODE_WORD_RE = re.compile(r"\b(quality|accurate|full)\b", re.IGNORECASE)


def parse_model_mode_request(text: str) -> str | None:
    """Classify a model-mode utterance.

    Returns:
        "fast"    — switch to fast mode
        "quality" — switch to quality mode
        "status"  — report the current mode + model
        None      — not a model-mode request (route as AI question)
    """
    t = (text or "").strip()
    if not t:
        return None
    if _MODEL_MODE_STATUS_RE.search(t):
        return "status"
    if (
        _MODEL_MODE_SWITCH_RE.search(t)
        or _MODEL_MODE_USE_RE.search(t)
        or _MODEL_MODE_TO_RE.search(t)
    ):
        if _FAST_MODE_WORD_RE.search(t):
            return "fast"
        if _QUALITY_MODE_WORD_RE.search(t):
            return "quality"
    return None


def _build_command_patterns() -> List[str]:
    """
    Build the router's COMMAND_PATTERNS from the trusted vocabulary in
    commands/system_commands.py (apps, sites, folders) plus fixed
    patterns for time, date, screenshot, volume, status, lock, power.
    """
    from commands.system_commands import APP_NAMES, FOLDER_NAMES, SITE_NAMES

    def _alt(names) -> str:
        return "|".join(re.escape(n) for n in names)

    app_alt = _alt(APP_NAMES)
    site_alt = _alt(SITE_NAMES)
    folder_alt = _alt(FOLDER_NAMES)

    return [
        # open/launch/start + any short noun phrase. The registry
        # resolves it against the trusted WEBSITES/APPS/FOLDERS tables
        # (with a safe fuzzy fallback for speech-recognition
        # misspellings like "open chrom") and politely declines unknown
        # targets — never executing arbitrary text.
        r"\b(?:open|launch|start)\s+(?:the\s+|my\s+)?"
        r"[a-z0-9][a-z0-9 .\-]{0,29}",

        # Launch desktop applications (exact vocabulary, incl. "run")
        rf'\b(open|launch|start|run)\s+(the\s+)?({app_alt})\b',

        # Open websites
        rf'\b(open|go to|visit|navigate to|launch)\s+'
        rf'(the\s+)?({site_alt})\b',

        # Open folders
        rf'\b(open|launch|show)\s+(my\s+|the\s+)?({folder_alt})\s*(folder)?\b',

        # Time and date (read from system clock, not AI)
        r'\bwhat(?:\'s| is)?\s+(the\s+)?'
        r'(current\s+)?(time|date)\b',
        r'\btell me the (time|date)\b',
        r'\bwhat day is (it|today)\b',
        r'\btoday\'?s date\b',

        # Screenshot
        r'\b(take a |take )?screenshot\b',

        # Volume control
        r'\b(volume (up|down|mute|unmute))\b',
        r'\b(increase|raise|decrease|lower|turn (up|down))\s+'
        r'(the\s+)?volume\b',
        r'\bset\s+(the\s+)?volume\s+to\s+\d{1,3}\s*(percent|%)?\b',
        r'\b(mute|unmute)\s+(the\s+)?(volume|sound|audio)\b',

        # System status
        r'\b(system status|computer status|pc status|system info|'
        r'system information|system health|how is my computer|'
        r'how is the computer)\b',

        # Lock screen
        r'\block\s+(my\s+|the\s+)?(computer|pc|laptop|system|machine|screen)\b',

        # System power (shutdown/restart/sleep — confirmation handled by
        # the registry's CONFIRM permission)
        r'\b(shut ?down|power off|restart|reboot|sleep|hibernate)\s+'
        r'(my\s+|the\s+)?(computer|pc|system|laptop|machine)\b',
        r'\b(abort|cancel|stop)\s+shut ?down\b',

        # Media control (kept for compatibility; the registry politely
        # declines rather than pretending)
        r'\b(play|pause|next track|previous track)\b',
    ]


class IntentRouter:
    """
    Routes user input to the correct handler.
    Defaults to AI so no question ever goes unanswered.
    """

    # ── Exit phrases ──────────────────────────────────────────
    EXIT_PATTERNS: List[str] = [
        r'\b(exit|quit|goodbye|good bye|bye jarvis|'
        r'shut ?down jarvis|stop jarvis|close jarvis|terminate)\b',
        r'^\s*bye\s*$',
    ]

    # ── Memory clear phrases ──────────────────────────────────
    MEMORY_PATTERNS: List[str] = [
        r'\b(clear|reset|wipe|erase)\s+'
        r'(the\s+)?(memory|history|context|conversation)\b',
        r'\bforget\s+(everything|all|our conversation)\b',
        r'\bstart\s+(over|fresh|a new conversation)\b',
    ]

    # ── Stop speaking ─────────────────────────────────────────
    STOP_PATTERNS: List[str] = [
        r'\b(stop|halt|cancel|silence)\s+'
        r'(speaking|talking|the voice|speech|that|generation|generating)\b',
        r'\bbe quiet\b',
    ]

    # ── System command phrases ────────────────────────────────
    # The registry (commands/registry.py) is the single resolver: the
    # router only decides *whether* an utterance is command-like, and
    # the registry decides *which* trusted command runs — so the two
    # can never disagree about the vocabulary.
    COMMAND_PATTERNS: List[str] = _build_command_patterns()

    def __init__(self):
        self._exit_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.EXIT_PATTERNS
        ]
        self._memory_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.MEMORY_PATTERNS
        ]
        self._stop_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.STOP_PATTERNS
        ]
        self._command_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.COMMAND_PATTERNS
        ]
        from brain.classifier import QuestionClassifier

        self.classifier = QuestionClassifier()
        logger.info("Intent router initialized.")

    def route(self, user_input: str) -> Tuple[str, str]:
        """
        Classify user input.

        Args:
            user_input: Raw text from STT or keyboard

        Returns:
            (intent, cleaned_text)

        Invalid input (empty, whitespace-only, or longer than
        MAX_INPUT_CHARS) is rejected up front with (UNKNOWN, "") — it
        never reaches a command handler or the AI provider.
        """
        validated = validate_input(user_input)
        if not validated:
            return Intent.UNKNOWN, ""

        # Normalize wake-name misrecognitions ("lajad" -> "jarvis"),
        # keeping the raw transcription in the debug log.
        normalized = normalize_wake_name(user_input)
        if normalized != user_input:
            logger.debug(
                f"Wake-name normalization: {user_input!r} -> {normalized!r}"
            )

        text = normalized.strip().lower()

        # 1 — Exit has highest priority
        if self._matches(text, self._exit_re):
            logger.info(f"Intent EXIT: '{text}'")
            return Intent.EXIT, text

        # 2 — Memory management
        if self._matches(text, self._memory_re):
            logger.info(f"Intent CLEAR_MEMORY: '{text}'")
            return Intent.CLEAR_MEMORY, text

        # 3 — Runtime model-mode control (fast/quality switch, status)
        mode_request = parse_model_mode_request(text)
        if mode_request is not None:
            logger.info(f"Intent MODEL_MODE: '{text}' -> {mode_request}")
            return Intent.MODEL_MODE, mode_request

        # 4 — Instant canned replies to greetings (no AI round-trip)
        if ENABLE_FAST_RESPONSES:
            reply = FAST_RESPONSES.get(_normalize_key(text))
            if reply:
                logger.info(f"Intent FAST_RESPONSE: '{text}'")
                return Intent.FAST_RESPONSE, reply

        # 5 — Stop speaking (interrupt TTS)
        if self._matches(text, self._stop_re):
            logger.info(f"Intent STOP_SPEECH: '{text}'")
            return Intent.STOP_SPEECH, text

        # 6 — System commands
        if self._matches(text, self._command_re):
            logger.info(f"Intent COMMAND: '{text}'")
            return Intent.COMMAND, text

        # 7 — Question classifier: current information needs a web search
        decision = self.classifier.classify(text)
        if decision == "web_search":
            logger.info(f"Intent WEB_SEARCH: '{text}'")
            return Intent.WEB_SEARCH, text

        # 8 — Deterministic clock/calendar the command patterns missed
        #     ("what day is today", "what is today date", ...)
        if decision in ("time_tool", "date_tool"):
            logger.info(f"Intent COMMAND (clock): '{text}'")
            return Intent.COMMAND, text

        # 9 — DEFAULT: send everything else to the AI brain
        logger.info(f"Intent AI_QUESTION: '{text}'")
        return Intent.AI_QUESTION, text

    def _matches(
        self,
        text: str,
        patterns: List[Pattern]
    ) -> bool:
        """Return True if text matches any compiled pattern."""
        return any(p.search(text) for p in patterns)
