"""
brain/classifier.py — Question classifier: does this need fresh information?

Runs on every non-command utterance and decides whether JARVIS must
look up current facts (WEB_SEARCH) or can answer from the local LLM's
knowledge (LOCAL_LLM).

Three modes, selected with AI_MODE in .env:

    auto  — decide per question (default)
    local — always use the local LLM (never search)
    web   — always search before answering

AUTO mode is a lightweight, deterministic scoring classifier (no ML
dependencies, no extra latency):

  * strong time-sensitivity markers ("current", "latest", "today",
    "recently", ...)  → +2
  * phrases like "what happened", "any news"  → +3
  * office-holder questions ("who is the prime minister")  → +3
  * topic signals for things that change (news, weather, prices,
    politics, sport, software versions, ...)  → +1

score >= 2 → WEB_SEARCH. Historical questions (past tense, explicit
years) are exempt so "who was the first president of India" stays local.

The classifier is deliberately biased toward LOCAL answers: an ordinary
educational question never triggers a web round-trip.
"""

import re

from commands.time_commands import DATE_RE, TIME_RE
from config import jarvis_config
from utils.logger import get_logger

logger = get_logger("classifier")

__all__ = [
    "QuestionClassifier",
    "TIME_TOOL",
    "DATE_TOOL",
    "WEB_SEARCH",
    "LOCAL_LLM",
]

# Classifier outputs.
TIME_TOOL = "time_tool"
DATE_TOOL = "date_tool"
WEB_SEARCH = "web_search"
LOCAL_LLM = "local_llm"

# ── Config (config.py is always import-safe; no local fallbacks) ─────────────
DEFAULT_MODE = jarvis_config.AI_MODE

_VALID_MODES = ("auto", "local", "web")

# Strong markers: the question is *about* the present moment.
_STRONG_RE = re.compile(
    r"\b(current|latest|recent|recently|today|tonight|"
    r"yesterday|tomorrow|updated|upcoming|breaking|fresh|newly|"
    r"right now|as of now|at present|this (week|month|year|morning|"
    r"afternoon|evening))\b",
    re.IGNORECASE,
)

# "electric/direct/alternating current" is physics, not the present day.
_CURRENT_PHYSICS_RE = re.compile(
    r"\b(electric|alternating|direct|eddy)\s+current\b",
    re.IGNORECASE,
)

# Phrases that ask for a rundown of recent events / news.
_PHRASE_RE = re.compile(
    r"\b(what happened|what's happening|what is happening|whats happening|"
    r"what has been happening|any news|in the news|is there any news|"
    r"tell me the news|today's news|the news today|news headlines|"
    r"what's new|whats new)\b",
    re.IGNORECASE,
)

# Office-holder questions are current by nature ("who is the CEO of X").
_OFFICE_HOLDER_RE = re.compile(
    r"\b(who is|who's|whos|who are|who're)\s+(the\s+|our\s+|new\s+)?"
    r"(current\s+|present\s+)?(ceo|president|prime minister|chief minister|"
    r"minister|chairman|chairperson|governor|mayor|secretary general|"
    r"director general|chancellor|premier|head of state|head of government)\b",
    re.IGNORECASE,
)

# Topics whose facts change over time.
_TOPIC_RE = re.compile(
    r"\b(news|weather|forecast|temperature|price|prices|stock|stocks|"
    r"market|bitcoin|crypto|cryptocurrency|election|elections|minister|"
    r"president|government|scheme|schemes|ranking|rankings|champion|"
    r"winner|won|match|score|scores|version|release|released|update|"
    r"updates|salary|gold rate|petrol|diesel|population|census|budget|"
    r"policy|law|legislation|bill|appointed|appointment|chairman|ceo|"
    r"team|player|fixture|result|results|incident|accident|attack|war|"
    r"peace|agreement|summit|poll|opinion poll|olympics|world cup|"
    r"tournament|championship|fixtures|standings|points table)\b",
    re.IGNORECASE,
)

# Past / historical framing: the answer does not need fresh facts.
_HISTORICAL_RE = re.compile(
    r"\b(was|were|had been|historically|during|before|back in|originally|"
    r"founded in|in (19|20)\d\d|the (19|20)\d\d)\b|"
    r"\b(19|20)\d\d\b",
    re.IGNORECASE,
)

# Greetings that happen to contain strong markers ("how are you today").
_GREETING_RE = re.compile(
    r"\b(how are you|how's it going|hows it going|how is it going|"
    r"how is your day|how are you doing|whats up|what's up)\b",
    re.IGNORECASE,
)


class QuestionClassifier:
    """Decides WEB_SEARCH vs LOCAL_LLM (and the deterministic time/date
    tools) for a non-command utterance."""

    def __init__(self, mode: str | None = None):
        mode = (mode or DEFAULT_MODE or "auto").strip().lower()
        if mode not in _VALID_MODES:
            logger.warning(
                f"Unknown AI_MODE '{mode}'; falling back to 'auto'."
            )
            mode = "auto"
        self.mode = mode

    def classify(self, text: str) -> str:
        """Return TIME_TOOL / DATE_TOOL / WEB_SEARCH / LOCAL_LLM."""
        t = (text or "").strip()
        if not t:
            return LOCAL_LLM

        # Deterministic clock/calendar — never ask the LLM for these.
        if TIME_RE.search(t):
            return TIME_TOOL
        if DATE_RE.search(t):
            return DATE_TOOL

        if self.mode == "local":
            return LOCAL_LLM
        if self.mode == "web":
            return WEB_SEARCH

        if _GREETING_RE.search(t):
            return LOCAL_LLM

        historical = bool(_HISTORICAL_RE.search(t))

        score = 0
        strong = _STRONG_RE.search(t)
        # "electric current" matches the strong word "current" but is
        # a physics topic, not a request for fresh information.
        if strong and not (
            strong.group(0).lower() == "current"
            and _CURRENT_PHYSICS_RE.search(t)
        ):
            score += 2
        if not historical:
            if _PHRASE_RE.search(t):
                score += 3
            if _OFFICE_HOLDER_RE.search(t):
                score += 3
            if _TOPIC_RE.search(t):
                score += 1

        decision = WEB_SEARCH if score >= 2 else LOCAL_LLM
        logger.debug(
            f"Classify (mode={self.mode}, score={score}, "
            f"historical={historical}): {t[:60]!r} -> {decision}"
        )
        return decision
