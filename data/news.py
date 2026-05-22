"""News & corporate-announcement ingestion.

Defines the news item shape and provider interface the sentiment engine
consumes. Concrete sources - NSE/BSE announcement feeds + RSS (free),
then NewsAPI / Finnhub - are wired up in Milestone 5, with retry and
rate-limit handling per source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class NewsItem:
    """A single headline or corporate announcement mapped to a symbol."""

    symbol: str
    headline: str
    source: str
    url: str = ""
    published: datetime | None = None
    is_announcement: bool = False        # exchange filing vs general news
    is_material: bool = False            # results / order / rating / M&A flag
    sentiment: float = 0.0               # -1..+1, filled by the sentiment engine

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return d


class NewsProvider:
    """Interface for news/announcement sources.

    Implementations: RSSNews (NSE/BSE + Moneycontrol/ET, free),
    NewsAPINews, FinnhubNews - delivered in Milestone 5.
    """

    name: str = "base"

    def fetch(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raise NotImplementedError("news ingestion lands in Milestone 5")


def get_provider(settings) -> NewsProvider:
    """Factory dispatching on settings.news_provider (Milestone 5)."""
    raise NotImplementedError(
        "news provider wiring lands in Milestone 5 "
        f"(requested provider: {settings.news_provider!r})"
    )
