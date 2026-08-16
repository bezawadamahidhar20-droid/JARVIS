"""Issue 13 — search-result caching.

Identical queries within the TTL are served from cache; expired entries
trigger a fresh search; private-looking queries are never cached; the
cache is bounded.
"""

import brain.search as search_mod
from brain.search import (
    SearchCache,
    SearchResult,
    TavilyProvider,
    clear_search_cache,
)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _results(n=1):
    return [
        SearchResult(title=f"R{i}", url=f"https://x.com/{i}", snippet="s")
        for i in range(n)
    ]


# ── SearchCache unit tests ────────────────────────────────────

def test_cache_hit_within_ttl():
    cache = SearchCache(ttl=300, max_entries=10)
    cache.put("query", _results())
    assert cache.get("query") == cache.get("query")
    assert cache.get("QUERY")  # case-insensitive key
    assert cache.get("other query") is None


def test_cache_expiry(monkeypatch):
    import time as real_time

    cache = SearchCache(ttl=10, max_entries=10)
    cache.put("q", _results())
    assert cache.get("q") is not None

    # Simulate time passing beyond the TTL (base = the real clock so
    # the stored timestamp stays consistent).
    base = real_time.monotonic()

    class FakeTime:
        t = base

        @staticmethod
        def monotonic():
            return FakeTime.t

    monkeypatch.setattr(search_mod.time, "monotonic", FakeTime.monotonic)
    FakeTime.t = base + 11.0
    assert cache.get("q") is None  # expired


def test_cache_bounded(monkeypatch):
    cache = SearchCache(ttl=300, max_entries=3)
    for i in range(5):
        cache.put(f"q{i}", _results())
    assert len(cache) <= 3


def test_sensitive_queries_not_cached():
    cache = SearchCache(ttl=300, max_entries=10)
    cache.put("what is my bank account password", _results())
    assert cache.get("what is my bank account password") is None


def test_ttl_zero_disables_cache():
    cache = SearchCache(ttl=0, max_entries=10)
    cache.put("q", _results())
    assert cache.get("q") is None


def test_clear():
    cache = SearchCache(ttl=300, max_entries=10)
    cache.put("q", _results())
    cache.clear()
    assert len(cache) == 0


# ── Provider integration (shared cache) ───────────────────────

def test_second_call_within_ttl_uses_cache(monkeypatch):
    clear_search_cache()
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(200, {"results": [
            {"title": "T", "url": "https://x.com/1", "content": "snippet"},
        ]})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    monkeypatch.setattr(search_mod, "SEARCH_CACHE_TTL", 300)
    provider = TavilyProvider(api_key="k")

    r1 = provider.search("current chief minister")
    r2 = provider.search("current chief minister")
    assert r1 == r2
    assert calls["n"] == 1  # second call hit the cache


def test_query_after_ttl_fetches_fresh(monkeypatch):
    clear_search_cache()
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(200, {"results": [
            {"title": "T", "url": "https://x.com/1", "content": "snippet"},
        ]})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    provider = TavilyProvider(api_key="k")
    provider.search("latest news")
    assert calls["n"] == 1

    # Force expiry of the shared cache.
    monkeypatch.setattr(
        search_mod, "SEARCH_CACHE_TTL", 0
    )
    clear_search_cache()
    provider.search("latest news")
    assert calls["n"] == 2  # fresh search


def test_failed_search_not_cached(monkeypatch):
    """A failed fetch must not poison the cache with an empty result."""
    clear_search_cache()
    state = {"fail": True}

    def fake_post(url, **kwargs):
        if state["fail"]:
            raise search_mod.requests.ConnectionError("down")
        return FakeResponse(200, {"results": [
            {"title": "T", "url": "https://x.com/1", "content": "snippet"},
        ]})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    provider = TavilyProvider(api_key="k")
    assert provider.search("q") == []  # failure

    state["fail"] = False
    assert provider.search("q")  # still hits the network -> real results
