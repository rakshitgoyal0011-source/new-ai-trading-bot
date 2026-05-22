"""Fundamental analysis engine  (Milestone 4).

Scores valuation, profitability, growth and financial health - SECTOR-
RELATIVE, i.e. a stock is graded against its peers, not absolute cutoffs
- into a FUNDAMENTAL SCORE (0-100) with reasons.

The result shape is fixed here so the composite engine and UI can be
built against it; the scoring logic is delivered in Milestone 4 once the
fundamentals provider (data/fundamentals.py) is wired up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FundamentalResult:
    symbol: str
    score: float                              # 0-100
    bias: str                                 # bullish | bearish | neutral
    reasons: list[str] = field(default_factory=list)
    sub_scores: dict[str, float] = field(default_factory=dict)  # valuation/...
    quality_flags: list[str] = field(default_factory=list)
    peers_compared: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class FundamentalEngine:
    """Sector-relative fundamental scorer (logic lands in Milestone 4)."""

    def analyze(self, symbol: str, data=None, peers=None) -> FundamentalResult:
        raise NotImplementedError(
            "fundamental scoring is implemented in Milestone 4 "
            "(needs the fundamentals provider + sector peer data)"
        )
