"""OPTIONAL Tavily integration test — hits the REAL Tavily API.

This test is skipped by default so `pytest` never consumes API credits.
Run it explicitly (from the repository root, with the key in .env):

    TAVILY_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_search_integration.py -v

It verifies the complete configured pipeline against the live API:
the factory builds a provider from .env and the key actually works.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TAVILY_INTEGRATION") != "1",
    reason="real-API test — set TAVILY_INTEGRATION=1 to run",
)


def test_live_tavily_search_returns_results():
    from brain.search import create_search_provider, filter_and_rank

    provider = create_search_provider("tavily")
    assert provider is not None
    assert provider.is_configured(), (
        "SEARCH_API_KEY is empty — set it in .env before running "
        "this integration test."
    )

    # The key must actually authenticate.
    assert provider.is_reachable() == "ok", (
        f"Tavily reachability = {provider.is_reachable()!r} — "
        "check that SEARCH_API_KEY in .env is valid."
    )

    results = provider.search("who is the current chief minister of andhra pradesh", max_results=3)
    assert results, "live search returned no results"

    ranked = filter_and_rank(results, "current chief minister")
    assert ranked, "no results survived ranking"
    assert ranked[0].title or ranked[0].snippet
    assert ranked[0].url, "top result missing a URL"


def test_live_tavily_rejects_bad_key():
    from brain.search import TavilyProvider

    provider = TavilyProvider(api_key="tvly-invalid-key-for-testing")
    # The API must reject the bad key — not treat it as a network blip.
    assert provider.is_reachable() == "auth"
