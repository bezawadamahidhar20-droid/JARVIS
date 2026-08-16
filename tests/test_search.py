"""Web search provider tests — all HTTP mocked, no real API calls."""

import brain.search as search_mod
from brain.search import (
    BraveProvider,
    SearchResult,
    SerperProvider,
    TavilyProvider,
    build_search_query,
    create_search_provider,
    filter_and_rank,
    format_results_for_llm,
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


def _patch_requests(monkeypatch, response, method="post"):
    monkeypatch.setattr(
        search_mod.requests, method, lambda *a, **k: response
    )


# ── Factory / config ──────────────────────────────────────────

def test_create_provider_defaults(monkeypatch):
    monkeypatch.setattr(search_mod, "SEARCH_PROVIDER", "tavily")
    provider = create_search_provider()
    assert isinstance(provider, TavilyProvider)


def test_create_provider_named():
    assert isinstance(create_search_provider("tavily"), TavilyProvider)
    assert isinstance(create_search_provider("serper"), SerperProvider)
    assert isinstance(create_search_provider("brave"), BraveProvider)


def test_create_provider_disabled_returns_none():
    for name in ("", "none", "disabled", "off", "local"):
        assert create_search_provider(name) is None


def test_create_provider_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        create_search_provider("not-a-provider")


def test_factory_passes_configured_key(monkeypatch):
    """The configured SEARCH_API_KEY must reach the provider — the
    factory used to instantiate providers with an empty key, so web
    search was always "not configured" even with a valid .env."""
    monkeypatch.setattr(search_mod, "SEARCH_API_KEY", "tvly-test-key")
    provider = create_search_provider("tavily")
    assert provider.is_configured() is True


def test_factory_missing_key_leaves_provider_unconfigured(monkeypatch):
    monkeypatch.setattr(search_mod, "SEARCH_API_KEY", "")
    provider = create_search_provider("tavily")
    assert provider.is_configured() is False


def test_factory_default_provider_uses_configured_key(monkeypatch):
    monkeypatch.setattr(search_mod, "SEARCH_PROVIDER", "serper")
    monkeypatch.setattr(search_mod, "SEARCH_API_KEY", "serper-key")
    provider = create_search_provider()
    assert isinstance(provider, SerperProvider)
    assert provider.is_configured() is True


# ── is_configured ─────────────────────────────────────────────

def test_not_configured_without_key():
    p = TavilyProvider(api_key="")
    assert p.is_configured() is False


def test_configured_with_key():
    p = TavilyProvider(api_key="secret")
    assert p.is_configured() is True


# ── Reachability (used by `jarvis --doctor`) ──────────────────

def test_is_reachable_unconfigured():
    assert TavilyProvider(api_key="").is_reachable() == "unconfigured"
    assert SerperProvider(api_key="").is_reachable() == "unconfigured"
    assert BraveProvider(api_key="").is_reachable() == "unconfigured"


def test_is_reachable_ok(monkeypatch):
    # Tavily/Serper use POST, Brave uses GET — mock both.
    _patch_requests(monkeypatch, FakeResponse(200, {}), method="post")
    _patch_requests(monkeypatch, FakeResponse(200, {}), method="get")
    assert TavilyProvider(api_key="k").is_reachable() == "ok"
    assert SerperProvider(api_key="k").is_reachable() == "ok"
    assert BraveProvider(api_key="k").is_reachable() == "ok"


def test_is_reachable_auth_failure(monkeypatch):
    """A rejected key (401/403) must be reported as an auth failure,
    not as a generic network problem."""
    for code in (401, 403):
        _patch_requests(monkeypatch, FakeResponse(code, {"error": "bad key"}), method="post")
        _patch_requests(monkeypatch, FakeResponse(code, {"error": "bad key"}), method="get")
        assert TavilyProvider(api_key="bad").is_reachable() == "auth"
        assert SerperProvider(api_key="bad").is_reachable() == "auth"
        assert BraveProvider(api_key="bad").is_reachable() == "auth"


def test_is_reachable_network_error(monkeypatch):
    def boom(*a, **k):
        raise search_mod.requests.ConnectionError("down")

    monkeypatch.setattr(search_mod.requests, "post", boom)
    monkeypatch.setattr(search_mod.requests, "get", boom)
    assert TavilyProvider(api_key="k").is_reachable() == "network"
    assert SerperProvider(api_key="k").is_reachable() == "network"
    assert BraveProvider(api_key="k").is_reachable() == "network"


def test_is_reachable_server_error_is_temporary(monkeypatch):
    """5xx / 429 are temporary unavailability, not a key problem."""
    _patch_requests(monkeypatch, FakeResponse(503, {}))
    assert TavilyProvider(api_key="k").is_reachable() == "network"


# ── Tavily ────────────────────────────────────────────────────

def test_tavily_parses_results(monkeypatch):
    _patch_requests(
        monkeypatch,
        FakeResponse(200, {
            "results": [
                {"title": "CM Office", "url": "https://example.gov.in/news",
                 "content": "The current Chief Minister is Naidu."},
                {"title": "No URL", "url": "", "content": "stub"},
            ]
        }),
    )
    results = TavilyProvider(api_key="k").search("current cm")
    assert len(results) == 2
    assert results[0].title == "CM Office"
    assert results[0].url == "https://example.gov.in/news"
    assert results[0].source == "example.gov.in"  # derived from URL
    assert "Naidu" in results[0].snippet


def test_tavily_unconfigured_returns_empty(monkeypatch):
    results = TavilyProvider(api_key="").search("anything")
    assert results == []


def test_tavily_failure_returns_empty(monkeypatch):
    _patch_requests(
        monkeypatch, FakeResponse(500, {"error": "boom"})
    )
    results = TavilyProvider(api_key="k").search("anything")
    assert results == []


def test_tavily_network_error_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise search_mod.requests.ConnectionError("down")

    monkeypatch.setattr(search_mod.requests, "post", boom)
    assert TavilyProvider(api_key="k").search("q") == []


def test_tavily_timeout_returns_empty(monkeypatch):
    def slow(*a, **k):
        raise search_mod.requests.Timeout("took too long")

    monkeypatch.setattr(search_mod.requests, "post", slow)
    assert TavilyProvider(api_key="k").search("q") == []
    assert TavilyProvider(api_key="k").is_reachable() == "network"


def test_tavily_sends_api_key_in_payload(monkeypatch):
    """The key travels in the request body — never in logs/console."""
    calls = {}

    def fake_post(url, **kwargs):
        calls.update(json=kwargs.get("json"))
        return FakeResponse(200, {"results": []})

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    TavilyProvider(api_key="tvly-secret").search("q")
    assert calls["json"]["api_key"] == "tvly-secret"
    assert calls["json"]["query"] == "q"


# ── Serper ────────────────────────────────────────────────────

def test_serper_parses_organic(monkeypatch):
    calls = {}

    def fake_post(url, **kwargs):
        calls.update(headers=kwargs.get("headers"), json=kwargs.get("json"))
        return FakeResponse(200, {
            "organic": [
                {"title": "News", "link": "https://news.example.com/a",
                 "snippet": "Latest update."},
            ]
        })

    monkeypatch.setattr(search_mod.requests, "post", fake_post)
    results = SerperProvider(api_key="k").search("latest news")
    assert len(results) == 1
    assert results[0].title == "News"
    assert results[0].url == "https://news.example.com/a"
    assert calls["headers"]["X-API-KEY"] == "k"


# ── Brave ─────────────────────────────────────────────────────

def test_brave_parses_web_results(monkeypatch):
    calls = {}

    def fake_get(url, **kwargs):
        calls.update(headers=kwargs.get("headers"), params=kwargs.get("params"))
        return FakeResponse(200, {
            "web": {
                "results": [
                    {"title": "Weather", "url": "https://weather.example.com",
                     "description": "Sunny today."},
                ]
            }
        })

    monkeypatch.setattr(search_mod.requests, "get", fake_get)
    results = BraveProvider(api_key="k").search("today's weather")
    assert len(results) == 1
    assert results[0].snippet == "Sunny today."
    assert calls["headers"]["X-Subscription-Token"] == "k"


# ── Result quality / ranking ──────────────────────────────────

def test_filter_and_rank_prefers_official_sources():
    results = [
        SearchResult(title="Spam", url="https://spam.example.com/x",
                     snippet="buy now"),
        SearchResult(title="Gov", url="https://www.india.gov.in/news",
                     snippet="official statement"),
        SearchResult(title="", url="", snippet=""),  # empty → dropped
    ]
    ranked = filter_and_rank(results, "current chief minister")
    assert len(ranked) == 2
    # Government source must rank above the spam page.
    assert ranked[0].url == "https://www.india.gov.in/news"


def test_filter_and_rank_drops_duplicates():
    results = [
        SearchResult(title="A", url="https://x.com/a", snippet="s"),
        SearchResult(title="B", url="https://x.com/a", snippet="different"),
    ]
    ranked = filter_and_rank(results, "query")
    assert len(ranked) == 1


def test_filter_and_rank_drops_empty():
    results = [SearchResult(title="", url="", snippet="")]
    assert filter_and_rank(results, "q") == []


def test_format_results_for_llm_includes_sources():
    text = format_results_for_llm([
        SearchResult(title="Title", url="https://x.com", snippet="Body."),
    ])
    assert "1. Title" in text
    assert "Body." in text
    assert "https://x.com" in text


def test_format_results_for_llm_truncates_long_snippets():
    """Long search snippets bloat the LLM prompt and slow prompt
    processing on CPU — they must be trimmed before going to Qwen3."""
    long_snippet = "word " * 400  # ~2000 chars
    text = format_results_for_llm([
        SearchResult(title="T", url="https://x.com", snippet=long_snippet),
    ])
    # The context stays compact: snippet is capped, URL is kept.
    assert len(text) < 400
    assert "https://x.com" in text
    assert "…" in text


def test_format_results_for_llm_caps_result_count():
    results = [
        SearchResult(title=f"R{i}", url=f"https://x.com/{i}", snippet="s")
        for i in range(10)
    ]
    text = format_results_for_llm(results)
    assert text.count("Source: https://x.com/") <= 5


def test_search_result_source_derived_from_url():
    r = SearchResult(title="t", url="https://www.example.com/page")
    assert r.source == "example.com"


def test_build_search_query_strips_fillers():
    assert build_search_query("so jarvis can you tell me the latest news") == "latest news"
    assert build_search_query("who is the current chief minister of andhra pradesh") == (
        "who is the current chief minister of andhra pradesh"
    )
    assert build_search_query("hey jarvis what happened today") == "what happened today"
    assert build_search_query("") == ""
