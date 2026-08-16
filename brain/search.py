"""
brain/search.py — Web search provider abstraction (current information)

When the question classifier decides an utterance needs *fresh* facts
(politics, news, weather, prices, sports, software versions, ...),
JARVIS searches the web and answers from the retrieved results instead
of trusting the model's training data.

Providers are pluggable and selected through .env:

    SEARCH_PROVIDER=tavily | serper | brave
    SEARCH_API_KEY=<your key>

Architecture:

    WebSearchProvider (ABC)
        ├── TavilyProvider   — POST api.tavily.com/search
        ├── SerperProvider   — POST google.serper.dev/search
        └── BraveProvider    — GET api.search.brave.com/res/v1/web/search

Search results are cached in memory (SearchCache) so identical queries
within SEARCH_CACHE_TTL seconds skip the external API entirely.

Design rules:
  * No API key is ever hardcoded — everything comes from config.py.
  * A provider that is not configured reports is_configured() == False
    and search() returns [] (never raises).
  * Search failures return [] — callers speak the honest
    "couldn't verify the latest information" message instead of
    hallucinating.
  * rank_results() prefers official / government / reputable sources
    and demotes obvious spam, so JARVIS does not blindly trust the
    first result.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from utils.logger import get_logger

logger = get_logger("search")

# ── Load config safely ────────────────────────────────────────
try:
    from config import search_config

    SEARCH_PROVIDER = search_config.PROVIDER
    SEARCH_API_KEY = search_config.API_KEY
    SEARCH_MAX_RESULTS = search_config.MAX_RESULTS
    SEARCH_CACHE_TTL = search_config.CACHE_TTL
    SEARCH_CACHE_MAX_ENTRIES = search_config.CACHE_MAX_ENTRIES
except Exception:
    SEARCH_PROVIDER = "tavily"
    SEARCH_API_KEY = ""
    SEARCH_MAX_RESULTS = 5
    SEARCH_CACHE_TTL = 300
    SEARCH_CACHE_MAX_ENTRIES = 50

# Queries that look personal/private are never cached.
_SENSITIVE_QUERY_RE = re.compile(
    r"\b(password|passwords|passwd|credit card|debit card|ssn|"
    r"social security|bank account|account number|pin|otp|api key|"
    r"secret|private key|my address|my phone|my email)\b",
    re.IGNORECASE,
)


@dataclass
class SearchResult:
    """One normalized web search hit."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            self.source = _domain_of(self.url)

    @property
    def is_empty(self) -> bool:
        """True when the result carries no usable content at all."""
        return not (self.title.strip() or self.snippet.strip())


def _domain_of(url: str) -> str:
    """Extract a readable domain from a URL ('' on failure)."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


# ── Result cache ──────────────────────────────────────────────

class SearchCache:
    """
    Tiny in-memory cache for identical search queries.

    * Entries expire after ``ttl`` seconds (0 disables the cache).
    * The cache is bounded: the oldest entry is evicted when
      ``max_entries`` is exceeded.
    * Personal-looking queries are never cached.
    * All failures are handled by the caller — this class never raises.
    """

    def __init__(self, ttl: float | None = None, max_entries: int | None = None):
        self.ttl = float(ttl if ttl is not None else SEARCH_CACHE_TTL)
        self.max_entries = int(
            max_entries if max_entries is not None else SEARCH_CACHE_MAX_ENTRIES
        )
        # key (query, max_results) -> (monotonic timestamp, results)
        self._data: dict[tuple[str, int], tuple[float, list]] = {}

    @staticmethod
    def _key(query: str, max_results: int) -> tuple[str, int]:
        return ((query or "").strip().lower(), int(max_results))

    @staticmethod
    def _is_sensitive(query: str) -> bool:
        return bool(_SENSITIVE_QUERY_RE.search(query or ""))

    def get(self, query: str, max_results: int = SEARCH_MAX_RESULTS):
        """Return cached results for *query*, or None on miss/expiry."""
        if self.ttl <= 0 or self._is_sensitive(query):
            return None
        key = self._key(query, max_results)
        item = self._data.get(key)
        if item is None:
            return None
        stored_at, results = item
        if time.monotonic() - stored_at > self.ttl:
            self._data.pop(key, None)
            return None
        return results

    def put(self, query: str, results: list, max_results: int = SEARCH_MAX_RESULTS) -> None:
        """Cache *results* for *query* (bounded, never raises)."""
        if self.ttl <= 0 or self._is_sensitive(query):
            return
        try:
            key = self._key(query, max_results)
            self._data[key] = (time.monotonic(), list(results))
            # Bounded growth: evict the oldest entry when over capacity.
            while len(self._data) > max(1, self.max_entries):
                oldest = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest]
        except Exception as e:
            logger.warning(f"Search cache write failed: {e}")

    def clear(self) -> None:
        """Drop all cached entries."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# ── Abstract provider ─────────────────────────────────────────

class WebSearchProvider(ABC):
    """Interface every search backend must implement."""

    name: str = "base"

    def __init__(self, api_key: str = "", max_results: int = SEARCH_MAX_RESULTS):
        self.api_key = (api_key or "").strip()
        self.max_results = max(1, int(max_results or SEARCH_MAX_RESULTS))

    def is_configured(self) -> bool:
        """True when an API key is present (provider is usable)."""
        return bool(self.api_key)

    @abstractmethod
    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Run *query* and return normalized results.

        Returns [] on any failure — never raises, never invents data.
        Implementations call ``self._run_cached`` so repeated queries
        skip the external API within the TTL.
        """

    def _run_cached(self, query: str, max_results, fetch) -> list[SearchResult]:
        """Serve *query* from the shared cache or run *fetch* and cache.

        Only successful results are cached; sensitive queries bypass the
        cache entirely. Cache failures degrade to a live search.
        """
        limit = max_results or self.max_results
        try:
            hit = _SEARCH_CACHE.get(query, limit)
            if hit is not None:
                logger.debug(f"Search cache hit for {query!r}.")
                return hit
        except Exception as e:
            logger.warning(f"Search cache lookup failed: {e}")
        try:
            results = fetch()
        except Exception as e:
            logger.error(f"Search fetch failed: {e}")
            return []
        if results:
            try:
                _SEARCH_CACHE.put(query, results, limit)
            except Exception as e:
                logger.warning(f"Search cache store failed: {e}")
        return results

    # Reachability statuses returned by is_reachable().
    REACH_OK = "ok"            # API answered 200 — key works
    REACH_AUTH = "auth"        # key rejected (HTTP 401/403)
    REACH_NETWORK = "network"  # temporary unavailability (timeout, DNS, 5xx)
    REACH_UNCONFIGURED = "unconfigured"  # no API key set

    def is_reachable(self) -> str:
        """Ping the API. Returns one of REACH_OK / REACH_AUTH /
        REACH_NETWORK / REACH_UNCONFIGURED — never raises, and never
        exposes the API key in any message."""
        if not self.is_configured():
            return self.REACH_UNCONFIGURED
        return self.REACH_NETWORK

    @staticmethod
    def _reach_status(resp) -> str:
        """Classify a probe response: 200 = ok, 401/403 = bad key,
        anything else (429, 5xx) = temporarily unavailable."""
        if resp.status_code == 200:
            return WebSearchProvider.REACH_OK
        if resp.status_code in (401, 403):
            return WebSearchProvider.REACH_AUTH
        return WebSearchProvider.REACH_NETWORK

    def describe(self) -> str:
        if self.is_configured():
            return f"{self.name} (configured)"
        return f"{self.name} (no API key)"


# ── Tavily ────────────────────────────────────────────────────

class TavilyProvider(WebSearchProvider):
    """Tavily — https://tavily.com (search API, free tier available)."""

    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"
    TIMEOUT = 10

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not self.is_configured():
            logger.warning("Tavily not configured (missing SEARCH_API_KEY).")
            return []

        def _fetch() -> list[SearchResult]:
            resp = requests.post(
                self.ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results or self.max_results,
                    "search_depth": "basic",
                },
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                SearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    snippet=str(item.get("content", "") or ""),
                )
                for item in data.get("results", [])
                if isinstance(item, dict)
            ]

        try:
            return self._run_cached(query, max_results, _fetch)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    def is_reachable(self) -> str:
        if not self.is_configured():
            return self.REACH_UNCONFIGURED
        try:
            resp = requests.post(
                self.ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": "test",
                    "max_results": 1,
                    "search_depth": "basic",
                },
                timeout=8,
            )
            return self._reach_status(resp)
        except Exception:
            return self.REACH_NETWORK


# ── Serper ────────────────────────────────────────────────────

class SerperProvider(WebSearchProvider):
    """Serper.dev — Google Search API wrapper (free tier available)."""

    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"
    TIMEOUT = 10

    def _headers(self) -> dict:
        return {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not self.is_configured():
            logger.warning("Serper not configured (missing SEARCH_API_KEY).")
            return []

        def _fetch() -> list[SearchResult]:
            resp = requests.post(
                self.ENDPOINT,
                headers=self._headers(),
                json={"q": query, "num": max_results or self.max_results},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                SearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("link", "") or ""),
                    snippet=str(item.get("snippet", "") or ""),
                )
                for item in data.get("organic", [])
                if isinstance(item, dict)
            ]

        try:
            return self._run_cached(query, max_results, _fetch)
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []

    def is_reachable(self) -> str:
        if not self.is_configured():
            return self.REACH_UNCONFIGURED
        try:
            resp = requests.post(
                self.ENDPOINT,
                headers=self._headers(),
                json={"q": "test", "num": 1},
                timeout=8,
            )
            return self._reach_status(resp)
        except Exception:
            return self.REACH_NETWORK


# ── Brave ─────────────────────────────────────────────────────

class BraveProvider(WebSearchProvider):
    """Brave Search API — https://brave.com/search/api/."""

    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    TIMEOUT = 10

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not self.is_configured():
            logger.warning("Brave not configured (missing SEARCH_API_KEY).")
            return []

        def _fetch() -> list[SearchResult]:
            resp = requests.get(
                self.ENDPOINT,
                headers={"X-Subscription-Token": self.api_key},
                params={"q": query, "count": max_results or self.max_results},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                SearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    snippet=str(item.get("description", "") or ""),
                )
                for item in data.get("web", {}).get("results", [])
                if isinstance(item, dict)
            ]

        try:
            return self._run_cached(query, max_results, _fetch)
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return []

    def is_reachable(self) -> str:
        if not self.is_configured():
            return self.REACH_UNCONFIGURED
        try:
            resp = requests.get(
                self.ENDPOINT,
                headers={"X-Subscription-Token": self.api_key},
                params={"q": "test", "count": 1},
                timeout=8,
            )
            return self._reach_status(resp)
        except Exception:
            return self.REACH_NETWORK


# ── Factory ───────────────────────────────────────────────────

_PROVIDERS: dict[str, type[WebSearchProvider]] = {
    "tavily": TavilyProvider,
    "serper": SerperProvider,
    "brave": BraveProvider,
}


# Shared cache used by every provider instance.
_SEARCH_CACHE = SearchCache()


def clear_search_cache() -> None:
    """Drop all cached search results (used by tests)."""
    _SEARCH_CACHE.clear()


def create_search_provider(name: str | None = None) -> WebSearchProvider | None:
    """Build the configured search provider.

    Args:
        name: provider key; defaults to SEARCH_PROVIDER from .env.

    Returns:
        WebSearchProvider instance, or None when search is disabled
        (SEARCH_PROVIDER empty / "none" / "disabled").

    Raises:
        ValueError: for an unknown provider key (callers catch this and
            continue with web search disabled).
    """
    if name is None:
        name = SEARCH_PROVIDER
    name = (name or "").strip().lower()
    if not name or name in ("none", "disabled", "off", "local"):
        return None
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown SEARCH_PROVIDER '{name}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))} (or leave empty to disable)"
        )
    # Pass the configured key explicitly — providers have no other way
    # to learn it (defaults to "" in the constructor).
    return _PROVIDERS[name](api_key=SEARCH_API_KEY)


# ── Result processing ─────────────────────────────────────────

# Domains trusted for factual / current-information answers.
_OFFICIAL_DOMAINS = (
    ".gov", ".gov.in", ".gov.uk", ".mil",
    "wikipedia.org",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "nytimes.com", "wsj.com", "ft.com",
    "economist.com", "aljazeera.com", "dw.com", "france24.com",
    "indianexpress.com", "thehindu.com", "ndtv.com",
    "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "pib.gov.in", "niti.gov.in", "eci.gov.in", "mea.gov.in",
    "pmo.gov.in", "whitehouse.gov", "state.gov", "europa.eu",
    "un.org", "who.int", "nasa.gov", "census.gov",
    "bloomberg.com", "cnbc.com", "marketwatch.com",
)

# Low-trust / SEO-heavy domains demoted for factual questions.
_SPAM_DOMAINS = (
    "quora.com", "reddit.com", "facebook.com", "instagram.com",
    "tiktok.com", "pinterest.com", "x.com", "twitter.com",
    "medium.com", "substack.com", "buzzfeed.com",
    "wattpad.com", "answers.com", "yahoo.answers",
    "jagran.com",  # heavy SEO aggregation
)


def _score_result(result: SearchResult, query: str) -> int:
    """Heuristic quality score for a search result (higher = better)."""
    score = 0
    domain = result.source.lower()

    if any(d in domain for d in _OFFICIAL_DOMAINS):
        score += 3
    if any(d in domain for d in _SPAM_DOMAINS):
        score -= 2

    # Results that mention the query keywords are usually more on-topic.
    keywords = [
        w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", query.lower())
        if w not in ("what", "who", "which", "where", "when", "how",
                     "the", "and", "is", "are", "of", "for", "in", "to",
                     "current", "latest", "today", "please")
    ]
    blob = f"{result.title} {result.snippet}".lower()
    matches = sum(1 for w in keywords if w in blob)
    score += min(2, matches)

    return score


def filter_and_rank(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Drop empty/duplicate results and sort by quality.

    Duplicates are detected by URL, keeping the first (highest-ranked
    by the search engine) occurrence.
    """
    seen: set[str] = set()
    clean: list[SearchResult] = []
    for r in results:
        if r.is_empty:
            continue
        key = r.url or r.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)

    clean.sort(key=lambda r: _score_result(r, query), reverse=True)
    return clean


# Long snippets are trimmed so the LLM prompt stays small — a huge
# context costs prompt-processing time on CPU and tempts the model to
# copy whole passages instead of answering concisely.
_MAX_SNIPPET_CHARS = 220


def format_results_for_llm(results: list[SearchResult], max_results: int = 5) -> str:
    """Render search results as compact text the LLM can answer from.

    The model is told to answer *only* from this verified information.
    Each snippet is truncated to ~220 chars so the prompt stays lean.
    """
    lines: list[str] = []
    for i, r in enumerate(results[:max_results], 1):
        lines.append(f"{i}. {r.title or 'Untitled'}")
        if r.snippet:
            snippet = r.snippet.strip()
            if len(snippet) > _MAX_SNIPPET_CHARS:
                snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "…"
            lines.append(f"   {snippet}")
        if r.url:
            lines.append(f"   Source: {r.url}")
    return "\n".join(lines)


def build_search_query(user_input: str) -> str:
    """Clean a user utterance into a usable search query.

    Strips the assistant's name, politeness fillers, and leading
    address phrases ("so", "can you tell me", ...). The raw question
    is otherwise preserved — search engines handle natural queries well.
    """
    text = (user_input or "").strip().lower()
    text = re.sub(r"\b(please|jarvis)\b", "", text)
    text = re.sub(r"^(so|hey|okay|ok|well|now|right)[,\s]+", "", text)
    text = re.sub(
        r"^(can|could|would|will|do|did|does|are)\s+(you|u)\s+",
        "", text,
    )
    text = re.sub(r"^(please\s+)?(tell me|let me know)\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    query = " ".join(text.split())
    return query or (user_input or "").strip()
