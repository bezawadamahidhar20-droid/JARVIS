"""
brain/classifier.py — Question classifier for web search vs local LLM.
 
[FIX m5] Added __all__ exports.
"""
 
import re
 
__all__ = [
    "QuestionClassifier",
]
 
# Keywords that indicate current/fresh information needed
_CURRENT_INFO_PATTERNS = [
    r"\b(current|latest|recent|today|now|this week|this month|this year)\b",
    r"\b(who is the|who's the)\s+(current|new|present)\b",
    r"\b(what is the|what's the)\s+(current|latest|today's)\b",
    r"\b(price|stock|weather|news|score|result)\b",
    r"\b(who won|who is winning|what happened)\b",
    r"\b(breaking|live|update|announcement)\b",
    r"\b(release date|when will|when does)\b.*\b(202[4-9]|203[0-9])\b",
    r"\b(president|prime minister|ceo|chairman)\s+of\b",
    r"\b(version|release)\s+of\s+\w+\s+(is|version)\b",
]
 
# Keywords that indicate static/timeless knowledge
_STATIC_PATTERNS = [
    r"\b(what is|define|explain|how does|why does)\b.*\b(work|mean|concept)\b",
    r"\b(history of|origin of|invented by)\b",
    r"\b(recipe for|how to make|how to cook)\b",
    r"\b(capital of|population of|located in)\b",
    r"\b(formula for|equation for|calculate)\b",
    r"\b(write|generate|create)\s+(a|an|the)?\s*(code|program|script)\b",
]
 
 
class QuestionClassifier:
    """Classifies questions as needing web search or local LLM."""
    
    def __init__(self):
        self._current_patterns = [
            re.compile(p, re.IGNORECASE) for p in _CURRENT_INFO_PATTERNS
        ]
        self._static_patterns = [
            re.compile(p, re.IGNORECASE) for p in _STATIC_PATTERNS
        ]
    
    def needs_search(self, text: str) -> bool:
        """Return True if the question likely needs fresh information."""
        if not text:
            return False
        
        # Check for current info patterns
        for pattern in self._current_patterns:
            if pattern.search(text):
                return True
        
        # If it matches static patterns, probably doesn't need search
        for pattern in self._static_patterns:
            if pattern.search(text):
                return False
        
        # Default: no search needed for most questions
        return False
    
    def classify(self, text: str) -> str:
        """Classify the question type."""
        if self.needs_search(text):
            return "web_search"
        return "local_llm"
 