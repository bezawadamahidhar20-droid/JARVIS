"""
brain/router.py — Intent Router
 
[FIX m7] Expanded wake word variants regex with more Whisper misrecognitions.
[FIX m5] Added __all__ exports.
[FIX m1] Removed try/except config fallbacks.
"""
 
import re
 
from config import jarvis_config
from utils.logger import get_logger
 
__all__ = [
    "Intent",
    "IntentRouter",
    "validate_input",
    "sanitize_input",
    "normalize_wake_name",
    "FAST_RESPONSES",
]
 
logger = get_logger("router")
 
 
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
 
 
OWNER = jarvis_config.OWNER
ENABLE_FAST_RESPONSES = jarvis_config.ENABLE_FAST_RESPONSES
MAX_INPUT_CHARS = jarvis_config.MAX_INPUT_CHARS
 
 
def sanitize_input(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())
 
 
def validate_input(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> str:
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
    "hi": "Hi there, {owner}.",
    "hi jarvis": "Hi, {owner}. How may I assist you?",
    "hey": "Hey, {owner}.",
    "hey jarvis": "Hey, {owner}. Ready when you are.",
    "good morning": "Good morning, {owner}.",
    "good afternoon": "Good afternoon, {owner}.",
    "good evening": "Good evening, {owner}.",
    "thanks": "You're welcome, {owner}.",
    "thank you": "You're welcome, {owner}.",
    "how are you": "I'm running at full capacity, {owner}. How can I help?",
    "who are you": "I am JARVIS, your personal AI assistant, {owner}.",
    "what is your name": "I am JARVIS, at your service, {owner}.",
}
 
 
def _normalize_key(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()
 
 
FAST_RESPONSES: dict = {
    key: reply.format(owner=OWNER)
    for key, reply in _FAST_TEMPLATES.items()
}
 
 
# [FIX m7] Expanded wake-name variants with more Whisper misrecognitions
_WAKE_VARIANTS_RE = re.compile(
    r"\b("
    r"jervis|lajav|lajad|jarivs|jerivs|"  # Original 5
    r"javas|jarves|jarvus|jarfis|gervis|garvis|djarvis|"  # Common misrecognitions
    r"charvis|jarvas|jarvs|jarvi|jervas|jarbis|"  # More variants
    r"jarves|jarwis|jarvice|jarves|larvus|jorvus"  # Additional
    r")\b",
    re.IGNORECASE,
)
 
_PAUSE_WORDS = (
    "so", "hey", "hi", "hello", "yo", "okay", "ok", "please",
    "now", "right", "well", "alright",
)
 
 
def _is_address_position(prefix: str) -> bool:
    p = prefix.strip()
    if not p:
        return True
    if p[-1] in ",;:!?.":
        return True
    last_word = re.sub(r"[^a-z']", "", p.split()[-1].lower()) if p.split() else ""
    return last_word in _PAUSE_WORDS
 
 
def normalize_wake_name(text: str) -> str:
    """Rewrite misrecognized variants of JARVIS to 'jarvis'."""
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
 
 
# Model mode patterns
_MODEL_MODE_SWITCH_RE = re.compile(
    r"\b(switch|change|set|use|go)\s+(to\s+|into\s+)?(the\s+)?"
    r"(fast|quick|speed|quality|accurate|full)\s+(mode|model|one)\b",
    re.IGNORECASE,
)
 
_MODEL_MODE_STATUS_RE = re.compile(
    r"\b(which|what)\s+model\s+(are you (using|running|on)|do you use)\b"
    r"|\bwhat\s+mode\s+(are you (in|using)|is this)\b",
    re.IGNORECASE,
)
 
 
def parse_model_mode(text: str) -> tuple[str, str] | None:
    """Parse model mode command. Returns (action, mode) or None."""
    if _MODEL_MODE_STATUS_RE.search(text):
        return ("status", "")
    
    match = _MODEL_MODE_SWITCH_RE.search(text)
    if match:
        mode_word = match.group(4).lower()
        if mode_word in ("fast", "quick", "speed"):
            return ("switch", "fast")
        elif mode_word in ("quality", "accurate", "full"):
            return ("switch", "quality")
    
    return None
 
 
class IntentRouter:
    """Routes user input to the appropriate handler."""
 
    def __init__(self):
        self._exit_re = re.compile(
            r"\b(goodbye|exit|quit|shutdown jarvis|stop jarvis|"
            r"terminate|bye|see you|good night jarvis)\b",
            re.IGNORECASE,
        )
        self._clear_re = re.compile(
            r"\b(clear|reset|wipe|erase|forget)\s+"
            r"(the\s+)?(memory|history|context|conversation|everything|all)\b"
            r"|\bstart\s+(over|fresh|a new conversation)\b",
            re.IGNORECASE,
        )
        self._stop_re = re.compile(
            r"\b(stop|shut up|be quiet|silence|enough|cancel)\b",
            re.IGNORECASE,
        )
 
    def route(self, text: str) -> tuple[str, str]:
        """
        Determine intent for user input.
        Returns (intent_type, optional_data).

        Invalid input (empty, whitespace-only, or longer than
        MAX_INPUT_CHARS) is rejected up front with (UNKNOWN, "") — it
        never reaches a command handler or the AI provider.
        """
        if not text or not text.strip():
            return (Intent.UNKNOWN, "")
        
        if len(text) > MAX_INPUT_CHARS:
            logger.warning(
                f"Input rejected: {len(text)} chars exceeds the "
                f"{MAX_INPUT_CHARS} char limit."
            )
            return (Intent.UNKNOWN, "")
        
        text = normalize_wake_name(text)
        normalized = _normalize_key(text)
        
        # Exit check
        if self._exit_re.search(text):
            return (Intent.EXIT, text)
        
        # Clear memory
        if self._clear_re.search(text):
            return (Intent.CLEAR_MEMORY, text)
        
        # Stop speech
        if self._stop_re.search(text) and len(normalized.split()) <= 3:
            return (Intent.STOP_SPEECH, text)
        
        # Model mode
        mode_result = parse_model_mode(text)
        if mode_result:
            return (Intent.MODEL_MODE, mode_result)
        
        # Fast responses (greetings)
        if ENABLE_FAST_RESPONSES:
            if normalized in FAST_RESPONSES:
                return (Intent.FAST_RESPONSE, FAST_RESPONSES[normalized])
        
        # Default: AI question (commands handled by registry)
        return (Intent.AI_QUESTION, text)
 