"""Pytest configuration — make the repository root importable."""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _reset_search_rate_limiter():
    """Isolate the module-level search rate limiter between tests.

    brain.search keeps a single shared SearchRateLimiter so the
    production process throttles real API calls. Without a reset,
    one test's searches would exhaust the window and make later
    tests (and even the same test twice) return [] unexpectedly.
    """
    yield
    try:
        from brain.search import reset_search_rate_limiter

        reset_search_rate_limiter()
    except Exception:
        pass
