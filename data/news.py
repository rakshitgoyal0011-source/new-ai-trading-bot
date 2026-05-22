"""News & corporate-announcement ingestion.

`NewsProvider` is the interface; two implementations ship today:
- `DemoNews`     synthetic, deterministic per symbol (no network)
- `RSSNews`      NSE/BSE announcements + Moneycontrol / ET headlines

The factory `get_provider(settings)` picks one based on DALAL_MODE
and NEWS_PROVIDER. Real RSS endpoints can throttle or change format,
so RSSNews fails soft (a dead feed never breaks ingestion).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np

from config import universe
from monitoring.logging import get_logger

log = get_logger("data.news")


@dataclass
class NewsItem:
    symbol: str
    headline: str
    source: str
    url: str = ""
    published: datetime | None = None
    is_announcement: bool = False
    is_material: bool = False
    sentiment: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d


class NewsProvider:
    name: str = "base"

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Demo (synthetic) news - always available, no network.
# --------------------------------------------------------------------------
_DEMO_TEMPLATES = [
    "{name} beats analyst estimates with strong quarterly results",
    "{name} announces new manufacturing facility expansion",
    "{name} raises dividend, signals robust profit outlook",
    "{name} bags large order, contract win lifts margins",
    "{name} stock surges after rating upgrade from Morgan Stanley",
    "{name} reports record quarterly revenue, beats guidance",
    "Brokerages turn bullish on {name} after Q2 earnings",
    "{name} faces SEBI investigation over disclosure norms",
    "{name} downgrades guidance amid weak demand environment",
    "{name} CFO resigns; shares fall on management concerns",
    "{name} hit by regulatory probe, stock plunges intraday",
    "Analysts cut price target on {name} citing slowing growth",
]


class DemoNews(NewsProvider):
    name = "demo"

    def fetch(self, symbol: str, limit: int = 5) -> list[NewsItem]:
        rng = np.random.default_rng(abs(hash("news_" + symbol)) % (2**31))
        stock = universe.get(symbol)
        name = stock.name if stock else symbol
        now = datetime.now()
        n = min(limit, int(rng.integers(3, 7)))
        idxs = rng.choice(len(_DEMO_TEMPLATES), size=n, replace=False)
        items: list[NewsItem] = []
        for i in idxs:
            days_old = float(rng.uniform(0.05, 6.5))
            headline = _DEMO_TEMPLATES[int(i)].format(name=name)
            items.append(NewsItem(
                symbol=symbol, headline=headline, source="demo wire",
                published=now - timedelta(days=days_old),
            ))
        return items


# --------------------------------------------------------------------------
# RSS news - NSE/BSE announcements + Moneycontrol + ET, free.
# --------------------------------------------------------------------------
_RSS_FEEDS: list[tuple[str, str, bool]] = [
    # (label, url, is_announcement)
    ("NSE Announcements", "https://www.nseindia.com/api/rss-corporate-announcements", True),
    ("Moneycontrol Markets", "https://www.moneycontrol.com/rss/business.xml", False),
    ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", False),
]


class RSSNews(NewsProvider):
    """Symbol-aware filter over generic financial RSS feeds.

    Headlines that mention either the trading symbol or the company name
    are kept. NSE and BSE corporate-announcement endpoints intermittently
    require a browser User-Agent and may rate-limit - the provider fails
    soft and logs at WARNING when a feed breaks.
    """

    name = "rss"

    def __init__(self, ttl_seconds: int = 600):
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}
        self._ttl = ttl_seconds

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        import time

        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._ttl:
            return cached[1][:limit]

        items = self._fetch_uncached(symbol, limit)
        self._cache[symbol] = (time.time(), items)
        return items

    def _fetch_uncached(self, symbol: str, limit: int) -> list[NewsItem]:
        try:
            import feedparser  # lazy, also keeps it optional in tests
        except ImportError:
            log.warning("feedparser not installed - RSSNews disabled")
            return []

        stock = universe.get(symbol)
        name = stock.name if stock else symbol
        needles = {symbol.lower(), name.lower(), name.split()[0].lower()}
        pattern = re.compile(r"|".join(re.escape(n) for n in needles), re.I)

        items: list[NewsItem] = []
        for label, url, is_ann in _RSS_FEEDS:
            try:
                parsed = feedparser.parse(url, request_headers={
                    "User-Agent": "Mozilla/5.0 (DalalTerminal)"})
                for entry in parsed.entries[:80]:
                    title = entry.get("title", "") or ""
                    if not pattern.search(title):
                        continue
                    pub = self._parse_date(entry.get("published_parsed"))
                    items.append(NewsItem(
                        symbol=symbol, headline=title.strip(),
                        source=label, url=entry.get("link", "") or "",
                        published=pub, is_announcement=is_ann,
                    ))
            except Exception as exc:
                log.warning("RSS feed %r failed: %s", label, exc)

        items.sort(key=lambda i: i.published or datetime.min, reverse=True)
        return items[:limit]

    @staticmethod
    def _parse_date(struct) -> datetime | None:
        if not struct:
            return None
        try:
            return datetime(*struct[:6])
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def get_provider(settings) -> NewsProvider:
    """Pick a news provider based on settings (demo / rss / ...)."""
    if not settings.is_live or settings.news_provider == "demo":
        return DemoNews()
    if settings.news_provider == "rss":
        return RSSNews()
    log.warning("unknown news provider %r - falling back to demo",
                settings.news_provider)
    return DemoNews()
