"""
brain/exceptions.py — Typed JARVIS exceptions.

Purpose: separate *expected, recoverable* failures (Ollama down, search
rate-limited, breaker open) from *genuine bugs*. Callers catch the
specific types and degrade gracefully; anything else is logged with a
full traceback and re-raised so it can never be silently swallowed.

Hierarchy:
    JARVISError
      ├─ ProviderUnavailableError   — AI provider unreachable/offline
      ├─ OllamaTimeoutError         — Ollama exceeded the request timeout
      ├─ CircuitOpenError           — circuit breaker fast-failed the call
      ├─ SearchError                — web search failed
      │    ├─ SearchTimeoutError
      │    └─ SearchRateLimitError
      └─ CommandError               — a registered command could not run
"""


class JARVISError(Exception):
    """Base class for all JARVIS-specific errors."""


class ProviderUnavailableError(JARVISError):
    """The AI provider is not reachable / not configured."""


class OllamaTimeoutError(JARVISError):
    """Ollama did not respond within the configured timeout."""


class CircuitOpenError(JARVISError):
    """A request was fast-failed because the circuit breaker is open."""


class SearchError(JARVISError):
    """A web search request failed."""


class SearchTimeoutError(SearchError):
    """The search provider timed out."""


class SearchRateLimitError(SearchError):
    """The search provider returned a rate-limit / quota error."""


class CommandError(JARVISError):
    """A registered command could not be executed."""
