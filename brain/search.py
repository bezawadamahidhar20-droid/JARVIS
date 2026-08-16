"""
brain/search.py — Web search provider abstraction
 
[FIX M5] Added threading.Lock() for thread-safe cache access
[FIX m3] Added rate limiting for API calls
[FIX m5] Added __all__ exports
[FIX m1] Removed try/except config fallbacks - config.py is always safe
"""
 
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse
 
import requests
 
from config import search_config
from utils.logger import get_logger
 
logger = get_logger("search")
 
__all__ = [
    "SearchResult",
    "SearchCache",
    "WebSearchProvider",
    "TavilyProvider",
    "SerperProvider",
    "BraveProvider",
    "create_search_provider",
    "clear_search_cache",
    "filter_and_rank",
    "format_results_for_llm",
    "build_search_query",
]
 
# Config values - direct import, no fallbacks needed
SEARCH_PROVIDER = search_config.PROVIDER
SEARCH_API_KEY = search_config.API_KEY
SEARCH_MAX_RESULTS = search_config.MAX_RESULTS
SEARCH_CACHE_TTL = search_config.CACHE_TTL
SEARCH_CACHE_MAX_ENTRIES = search_config.CACHE_MAX_ENTRIES
SEARCH_RATE_LIMIT = search_config.RATE_LIMIT
SEARCH_RATE_WINDOW = search_config.RATE_WINDOW
 
 
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
        return not (self.title.strip() or self.snippet.strip())
 
 
def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""
 
 
# [FIX m3] Rate limiter for API calls
class RateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: deque = deque()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        """Return True if a call is allowed, False if rate limited."""
        now = time.monotonic()
        with self._lock:
            # Remove calls outside the window
            while self._calls and self._calls[0] < now - self.window:
                self._calls.popleft()
            
            if len(self._calls) >= self.max_calls:
                return False
            
            self._calls.append(now)
            return True
    
    def wait_time(self) -> float:
        """Return seconds to wait before next call is allowed."""
        now = time.monotonic()
        with self._lock:
            while self._calls and self._calls[0] < now - self.window:
                self._calls.popleft()
            
            if len(self._calls) < self.max_calls:
                return 0.0
            
            return self._calls[0] + self.window - now
 
 
class SearchCache:
    """
    Thread-safe in-memory cache for search queries.
    
    [FIX M5] All methods now use self._lock for thread safety.
    """
 
    def __init__(self, ttl: float | None = None, max_entries: int | None = None):
        self.ttl = float(ttl if ttl is not None else SEARCH_CACHE_TTL)
        self.max_entries = int(
            max_entries if max_entries is not None else SEARCH_CACHE_MAX_ENTRIES
        )
        self._data: dict[tuple[str, int], tuple[float, list]] = {}
        self._lock = threading.Lock()  # [FIX M5] Thread safety
 
    @staticmethod
    def _key(query: str, max_results: int) -> tuple[str, int]:
        return ((query or "").strip().lower(), int(max_results))
 
    @staticmethod
    def _is_sensitive(query: str) -> bool:
        return bool(_SENSITIVE_QUERY_RE.search(query or ""))
 
    def get(self, query: str, max_results: int = SEARCH_MAX_RESULTS):
        """Return cached results or None on miss/expiry. Thread-safe."""
        if self.ttl <= 0:
            return None
        
        with self._lock:  # [FIX M5]
            key = self._key(query, max_results)
            entry = self._data.get(key)
            if entry is None:
                return None
            
            ts, results = entry
            if time.monotonic() - ts > self.ttl:
                self._data.pop(key, None)
                return None
            return results
 
    def put(self, query: str, results: list, max_results: int = SEARCH_MAX_RESULTS) -> None:
        """Cache results. Thread-safe."""
        if self.ttl <= 0:
            return
        if self._is_sensitive(query):
            return
        
        with self._lock:  # [FIX M5]
            try:
                key = self._key(query, max_results)
                self._data[key] = (time.monotonic(), results)
                
                # Evict oldest if over limit
                if len(self._data) > max(1, self.max_entries):
                    oldest = min(self._data, key=lambda k: self._data[k][0])
                    del self._data[oldest]
            except Exception as e:
                logger.warning(f"Search cache write failed: {e}")
 
    def clear(self) -> None:
        """Drop all cached entries. Thread-safe."""
        with self._lock:  # [FIX M5]
            self._data.clear()
 
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
 
 
class WebSearchProvider(ABC):
    """Interface every search backend must implement."""
    name: str = "base"
 
    def __init__(self, api_key: str = "", max_results: int = SEARCH_MAX_RESULTS):
        self.api_key = (api_key or "").strip()
        self.max_results = max(1, int(max_results or SEARCH_MAX_RESULTS))
        self._rate_limiter = RateLimiter(SEARCH_RATE_LIMIT, SEARCH_RATE_WINDOW)
 
    def is_configured(self) -> bool:
        return bool(self.api_key)
 
    @abstractmethod
    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        pass
 
    def _run_cached(self, query: str, max_results, fetch) -> list[SearchResult]:
        """Serve from cache or run fetch with rate limiting."""
        limit = max_results or self.max_results
        
        # Check cache first
        try:
            hit = _SEARCH_CACHE.get(query, limit)
            if hit is not None:
                logger.debug(f"Search cache hit for {query!r}.")
                return hit
        except Exception as e:
            logger.warning(f"Search cache lookup failed: {e}")
        
        # [FIX m3] Check rate limit before API call
        if not self._rate_limiter.allow():
            wait = self._rate_limiter.wait_time()
            logger.warning(f"Rate limited. Wait {wait:.1f}s before next search.")
            return []
        
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
 
    # Reachability constants
    REACH_OK = "ok"
    REACH_AUTH = "auth"
    REACH_NETWORK = "network"
    REACH_UNCONFIGURED = "unconfigured"
 
    def is_reachable(self) -> str:
        if not self.is_configured():
            return self.REACH_UNCONFIGURED
        return self.REACH_NETWORK
 
    @staticmethod
    def _reach_status(resp) -> str:
        if resp.status_code == 200:
            return WebSearchProvider.REACH_OK
        if resp.status_code in (401, 403):
            return WebSearchProvider.REACH_AUTH
        return WebSearchProvider.REACH_NETWORK
 
    def describe(self) -> str:
        if self.is_configured():
            return f"{self.name} (configured)"
        return f"{self.name} (no API key)"
 
 
class TavilyProvider(WebSearchProvider):
    """Tavily search provider."""
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
 
 
class SerperProvider(WebSearchProvider):
    """Serper.dev search provider."""
    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"
    TIMEOUT = 10
 
    def _headers(self) -> dict:
        return {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
 
    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not self.is_configured():
            logger.warning("Serper not configured.")
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
 
 
class BraveProvider(WebSearchProvider):
    """Brave Search provider."""
    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    TIMEOUT = 10
 
    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        if not self.is_configured():
            logger.warning("Brave not configured.")
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
 
 
_PROVIDERS: dict[str, type[WebSearchProvider]] = {
    "tavily": TavilyProvider,
    "serper": SerperProvider,
    "brave": BraveProvider,
}
 
_SEARCH_CACHE = SearchCache()
 
 
def clear_search_cache() -> None:
    _SEARCH_CACHE.clear()
 
 
def create_search_provider(name: str | None = None) -> WebSearchProvider | None:
    if name is None:
        name = SEARCH_PROVIDER
    name = (name or "").strip().lower()
    if not name or name in ("none", "disabled", "off", "local"):
        return None
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown SEARCH_PROVIDER '{name}'.")
    return _PROVIDERS[name](api_key=SEARCH_API_KEY)
 
 
# Result ranking
_OFFICIAL_DOMAINS = (
    ".gov", ".gov.in", ".gov.uk", ".mil", "wikipedia.org",
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
)
 
_SPAM_DOMAINS = (
    "quora.com", "reddit.com", "facebook.com", "medium.com",
)
 
 
def _score_result(result: SearchResult, query: str) -> int:
    score = 0
    domain = result.source.lower()
    if any(d in domain for d in _OFFICIAL_DOMAINS):
        score += 3
    if any(d in domain for d in _SPAM_DOMAINS):
        score -= 2
    return score
 
 
def filter_and_rank(results: list[SearchResult], query: str) -> list[SearchResult]:
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
 
 
_MAX_SNIPPET_CHARS = 220
 
 
def format_results_for_llm(results: list[SearchResult], max_results: int = 5) -> str:
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
    text = (user_input or "").strip().lower()
    text = re.sub(r"\b(please|jarvis)\b", "", text)
    text = re.sub(r"^(so|hey|okay|ok|well|now|right)[,\s]+", "", text)
    text = re.sub(r"^(can|could|would|will|do|did|does|are)\s+(you|u)\s+", "", text)
    text = re.sub(r"^(please\s+)?(tell me|let me know)\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    query = " ".join(text.split())
    return query or (user_input or "").strip()
 