"""News / sentiment engine  (Milestone 5).

Scores recent news and corporate announcements per stock into a
NEWS/SENTIMENT SCORE (0-100): sentiment per item (finance lexicon first,
FinBERT as a swappable upgrade), weighted by recency and source
reliability, with material events (results, orders, downgrades,
regulatory, M&A) flagged.

The result shape is fixed here so the composite engine and UI can be
built against it; scoring is delivered in Milestone 5 once news
ingestion (data/news.py) is wired up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SentimentResult:
    symbol: str
    score: float                              # 0-100
    bias: str                                 # bullish | bearish | neutral
    reasons: list[str] = field(default_factory=list)
    headline_count: int = 0
    material_events: list[str] = field(default_factory=list)
    model: str = "lexicon"                    # lexicon | finbert

    def to_dict(self) -> dict:
        return asdict(self)


class SentimentEngine:
    """News-sentiment scorer (logic lands in Milestone 5)."""

    def __init__(self, model: str = "lexicon"):
        self.model = model

    def analyze(self, symbol: str, items=None) -> SentimentResult:
        raise NotImplementedError(
            "news/sentiment scoring is implemented in Milestone 5 "
            "(needs news ingestion + the sentiment model)"
        )
