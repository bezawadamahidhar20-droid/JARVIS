"""Clock and calendar commands (pure functions, no side effects).

The compiled regexes live here — not in the router — so that
``brain.router`` and ``commands.registry`` both reuse the exact same
definitions and can never drift out of sync.
"""

import re
from datetime import datetime

# Phrases like "what time is it" / "tell me the time" / "time now".
TIME_RE = re.compile(
    r"\b(what('?s| is)? the time|current time|tell me the time|"
    r"time right now|what time is it|what time now|time now)\b",
    re.IGNORECASE,
)

# Phrases like "what's the date" / "what day is it" / "today's date".
# "today'?s? date" also catches Whisper's missing apostrophe variants.
DATE_RE = re.compile(
    r"\b(what('?s| is)? the date|current date|what day is (it|today)|"
    r"today'?s? date|tell me the date|what date is it|todays date)\b",
    re.IGNORECASE,
)


def get_current_time() -> str:
    """Return the current time, e.g. "It's 3:45 PM"."""
    now = datetime.now()
    formatted = now.strftime("%I:%M %p").lstrip("0")
    return f"It's {formatted}."


def get_current_date() -> str:
    """Return today's date, e.g. "Today is Monday, July 14, 2025"."""
    now = datetime.now()
    return f"Today is {now:%A, %B} {now.day}, {now.year}."