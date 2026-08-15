"""
brain/router.py — Intent Router

Decides whether user input is:
  1. A system COMMAND  (open app, time, date, screenshot)
  2. An AI QUESTION    (route to Qwen3 via Ollama)
  3. EXIT              (shut down JARVIS)
  4. CLEAR_MEMORY      (reset conversation context)
  5. FAST_RESPONSE     (instant canned reply to a greeting)

CRITICAL RULE:
Anything that is NOT matched as a command becomes an
AI_QUESTION. This is the fix that makes JARVIS actually
answer normal questions instead of saying
"I don't understand".
"""

import re
from typing import Tuple, List, Pattern
from utils.logger import get_logger

logger = get_logger("router")


class Intent:
    """Intent type constants."""
    COMMAND = "command"
    AI_QUESTION = "ai_question"
    CLEAR_MEMORY = "clear_memory"
    EXIT = "exit"
    FAST_RESPONSE = "fast_response"
    UNKNOWN = "unknown"


# ── Fast responses ────────────────────────────────────────────
# Greetings answered instantly from this local table instead of
# a full AI round-trip. Keys are normalized (lowercase, no punctuation).
try:
    from config import jarvis_config
    OWNER = jarvis_config.OWNER
    ENABLE_FAST_RESPONSES = jarvis_config.ENABLE_FAST_RESPONSES
except Exception:
    OWNER = "Sir"
    ENABLE_FAST_RESPONSES = True

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
    "how are you doing": "All systems nominal, {owner}.",
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


class IntentRouter:
    """
    Routes user input to the correct handler.
    Defaults to AI so no question ever goes unanswered.
    """

    # ── Exit phrases ──────────────────────────────────────────
    EXIT_PATTERNS: List[str] = [
        r'\b(exit|quit|goodbye|good bye|bye jarvis|'
        r'shut ?down jarvis|stop jarvis|terminate)\b',
        r'^\s*bye\s*$',
    ]

    # ── Memory clear phrases ──────────────────────────────────
    MEMORY_PATTERNS: List[str] = [
        r'\b(clear|reset|wipe|erase)\s+'
        r'(the\s+)?(memory|history|context|conversation)\b',
        r'\bforget\s+(everything|all|our conversation)\b',
        r'\bstart\s+(over|fresh|a new conversation)\b',
    ]

    # ── System command phrases ────────────────────────────────
    # Keep this list tight. When in doubt let the AI handle it.
    COMMAND_PATTERNS: List[str] = [
        # Launch desktop applications
        r'\b(open|launch|start|run)\s+'
        r'(chrome|firefox|edge|browser|notepad|calculator|'
        r'calc|paint|word|excel|powerpoint|vlc|spotify|'
        r'discord|whatsapp|telegram|steam|file explorer|'
        r'explorer|task manager|command prompt|cmd|'
        r'powershell|terminal|settings)\b',

        # Open websites
        r'\b(open|go to|visit|navigate to|launch)\s+'
        r'(youtube|google|github|stack ?overflow|reddit|'
        r'twitter|instagram|facebook|linkedin|netflix|'
        r'amazon|gmail|chatgpt)\b',

        # Time and date (read from system clock, not AI)
        r'\bwhat(?:\'s| is)?\s+(the\s+)?'
        r'(current\s+)?(time|date)\b',
        r'\btell me the (time|date)\b',
        r'\bwhat day is (it|today)\b',
        r'\btoday\'?s date\b',

        # Screenshot
        r'\b(take a |take )?screenshot\b',

        # Media and volume control
        r'\b(volume (up|down|mute|unmute))\b',
        r'\b(increase|decrease|set)\s+(volume|brightness)\b',
        r'\b(play|pause|next track|previous track)\b',

        # System power
        r'\b(shut ?down|restart|reboot|sleep|lock)\s+'
        r'(the\s+)?(computer|pc|system|laptop|machine)\b',
        r'\b(abort|cancel)\s+shut ?down\b',
    ]

    def __init__(self):
        self._exit_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.EXIT_PATTERNS
        ]
        self._memory_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.MEMORY_PATTERNS
        ]
        self._command_re: List[Pattern] = [
            re.compile(p, re.IGNORECASE)
            for p in self.COMMAND_PATTERNS
        ]
        logger.info("Intent router initialized.")

    def route(self, user_input: str) -> Tuple[str, str]:
        """
        Classify user input.

        Args:
            user_input: Raw text from STT or keyboard

        Returns:
            (intent, cleaned_text)
        """
        if not user_input or not user_input.strip():
            return Intent.UNKNOWN, ""

        text = user_input.strip().lower()

        # 1 — Exit has highest priority
        if self._matches(text, self._exit_re):
            logger.info(f"Intent EXIT: '{text}'")
            return Intent.EXIT, text

        # 2 — Memory management
        if self._matches(text, self._memory_re):
            logger.info(f"Intent CLEAR_MEMORY: '{text}'")
            return Intent.CLEAR_MEMORY, text

        # 3 — Instant canned replies to greetings (no AI round-trip)
        if ENABLE_FAST_RESPONSES:
            reply = FAST_RESPONSES.get(_normalize_key(text))
            if reply:
                logger.info(f"Intent FAST_RESPONSE: '{text}'")
                return Intent.FAST_RESPONSE, reply

        # 4 — System commands
        if self._matches(text, self._command_re):
            logger.info(f"Intent COMMAND: '{text}'")
            return Intent.COMMAND, text

        # 5 — DEFAULT: send everything else to the AI brain
        logger.info(f"Intent AI_QUESTION: '{text}'")
        return Intent.AI_QUESTION, text

    def _matches(
        self,
        text: str,
        patterns: List[Pattern]
    ) -> bool:
        """Return True if text matches any compiled pattern."""
        return any(p.search(text) for p in patterns)