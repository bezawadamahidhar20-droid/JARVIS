"""
commands/time_commands.py — Time and date commands.
 
[FIX m5] Added __all__ exports.
"""
 
import re
from datetime import datetime
 
__all__ = [
    "TIME_RE",
    "DATE_RE",
    "get_time",
    "get_date",
]
 
TIME_RE = re.compile(
    r"\b(what time is it|what's the time|current time|"
    r"tell me the time|time please)\b",
    re.IGNORECASE,
)
 
DATE_RE = re.compile(
    r"\b(what date is it|what's the date|today's date|"
    r"current date|what day is it|tell me the date)\b",
    re.IGNORECASE,
)
 
 
def get_time() -> str:
    """Return the current time as a spoken string."""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    # 12-hour format
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    
    if minute == 0:
        return f"It's {hour_12} o'clock {period}."
    elif minute == 30:
        return f"It's half past {hour_12} {period}."
    elif minute == 15:
        return f"It's quarter past {hour_12} {period}."
    elif minute == 45:
        return f"It's quarter to {(hour_12 % 12) + 1} {period}."
    else:
        return f"It's {hour_12}:{minute:02d} {period}."
 
 
def get_date() -> str:
    """Return the current date as a spoken string."""
    now = datetime.now()
    
    # Day suffix
    day = now.day
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    
    return now.strftime(f"Today is %A, %B {day}{suffix}, %Y.")
 