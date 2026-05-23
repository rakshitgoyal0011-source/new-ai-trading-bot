"""News / sentiment engine.

Scores recent news + corporate announcements into a NEWS / SENTIMENT
SCORE (0-100). Default model is a finance-lexicon scorer
(data/lexicon.py); FinBERT (`ProsusAI/finbert`) can be plugged in by
setting SENTIMENT_MODEL=finbert - the engine accepts a swappable
`scorer` callable.

Sentiment per item is in [-1, +1]; the aggregate weights items by
recency (exponential decay with a 14-day half-life) and gives a small
boost to exchange announcements + items containing material-event
keywords (results, orders, M&A, regulatory).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from data.lexicon import MATERIAL, NEGATIVE, POSITIVE

_WORD_RE = re.compile(r"\b[a-z]+\b")


def lexicon_score(text: str) -> float:
    """Return a -1..+1 sentiment score for a free-text headline."""
    if not text:
        return 0.0
    tokens = _WORD_RE.findall(text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def is_material(text: str) -> bool:
    """True if a headline mentions a material-event keyword."""
    if not text:
        return False
    return any(t in MATERIAL for t in _WORD_RE.findall(text.lower()))


@dataclass
class SentimentResult:
    symbol: str
    score: float                              # 0-100
    bias: str                                 # bullish | bearish | neutral
    reasons: list[str] = field(default_factory=list)
    headline_count: int = 0
    material_events: list[str] = field(default_factory=list)
    model: str = "lexicon"

    def to_dict(self) -> dict:
        return asdict(self)


class SentimentEngine:
    """Aggregate per-item sentiment into a recency-weighted 0-100 score."""

    HALF_LIFE_DAYS = 14.0

    def __init__(
        self,
        model: str = "lexicon",
        scorer: Callable[[str], float] | None = None,
    ):
        self.model = model
        if scorer is not None:
            self.scorer = scorer
        elif model == "finbert":
            from analysis.sentiment.finbert import finbert_scorer
            self.scorer = finbert_scorer
        else:
            self.scorer = lexicon_score

    def analyze(self, symbol: str, items=None) -> SentimentResult:
        items = list(items or [])
        if not items:
            return SentimentResult(
                symbol=symbol, score=50.0, bias="neutral",
                reasons=["no news in window"], model=self.model,
            )

        now = datetime.now()
        weighted_sum = 0.0
        weight_sum = 0.0
        material: list[str] = []
        ranked: list[tuple[float, str]] = []   # (contribution_abs, headline)

        for item in items:
            s = item.sentiment if item.sentiment else self.scorer(item.headline)
            material_flag = item.is_material or is_material(item.headline)
            if material_flag and item.headline not in material:
                material.append(item.headline)

            age_days = self._age_days(item.published, now)
            recency = math.exp(-math.log(2) * age_days / self.HALF_LIFE_DAYS)
            source_boost = 1.25 if item.is_announcement else 1.0
            material_boost = 1.15 if material_flag else 1.0
            w = recency * source_boost * material_boost

            weighted_sum += s * w
            weight_sum += w
            ranked.append((abs(s) * w, f"{'[+]' if s > 0 else '[-]' if s < 0 else '[ ]'} {item.headline}"))

        avg = weighted_sum / weight_sum if weight_sum else 0.0
        score = round(max(0.0, min(100.0, 50.0 + 50.0 * avg)), 1)
        bias = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"

        ranked.sort(reverse=True)
        reasons = [headline for _, headline in ranked[:5]]
        return SentimentResult(
            symbol=symbol, score=score, bias=bias, reasons=reasons,
            headline_count=len(items), material_events=material,
            model=self.model,
        )

    @staticmethod
    def _age_days(published, now: datetime) -> float:
        if published is None:
            return 7.0  # unknown timestamp -> treat as a week old
        try:
            delta = now - published
            return max(0.0, delta.total_seconds() / 86_400.0)
        except (TypeError, AttributeError):
            return 7.0
