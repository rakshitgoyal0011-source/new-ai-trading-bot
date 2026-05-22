"""Screeners, ranking and the TOP leaderboard.

`rank()` powers the `TOP` command - it orders the universe by composite
score with sector / price / score filters. Prebuilt scans that need only
technical data run today; scans that depend on fundamentals or news are
flagged for their milestone so the UI can say so honestly.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class LeaderboardEntry:
    """One ranked row: a stock plus the numbers behind its placement."""

    symbol: str
    name: str = ""
    sector: str = ""
    composite_score: float = 50.0
    technical_score: float = 50.0
    bias: str = "neutral"
    trend: str = "sideways"
    ltp: float = 0.0
    change_pct: float = 0.0
    atr: float = 0.0
    top_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def rank(
    entries: list[LeaderboardEntry],
    *,
    sector: str | None = None,
    min_score: float = 0.0,
    min_price: float | None = None,
    max_price: float | None = None,
    max_results: int = 20,
) -> list[LeaderboardEntry]:
    """Filter and order leaderboard entries by composite score (desc)."""
    rows = list(entries)
    if sector:
        rows = [e for e in rows if e.sector.lower() == sector.lower()]
    if min_score:
        rows = [e for e in rows if e.composite_score >= min_score]
    if min_price is not None:
        rows = [e for e in rows if e.ltp >= min_price]
    if max_price is not None:
        rows = [e for e in rows if e.ltp <= max_price]
    rows.sort(key=lambda e: e.composite_score, reverse=True)
    return rows[:max_results]


@dataclass(frozen=True)
class Scan:
    key: str
    label: str
    description: str
    ready: bool                                  # False -> pending a milestone
    predicate: Callable[[LeaderboardEntry], bool] | None = None


SCANS: list[Scan] = [
    Scan("momentum", "Momentum",
         "strong technical score in a confirmed uptrend", True,
         lambda e: e.technical_score >= 65 and e.trend == "uptrend"),
    Scan("oversold", "Oversold bounce",
         "RSI below 35 but technical structure still intact", True,
         lambda e: e.metrics.get("rsi", 50) < 35 and e.technical_score >= 45),
    Scan("volume_spike", "Volume spike",
         "traded volume at least 2x its 20-bar average", True,
         lambda e: e.metrics.get("volume_ratio", 1.0) >= 2.0),
    Scan("strong_bull", "Strong bullish",
         "composite score 70+ with a bullish bias", True,
         lambda e: e.composite_score >= 70),
    Scan("breakout_news", "Breakout + positive news",
         "breakout confirmed by positive sentiment - needs Milestone 5",
         False),
    Scan("value", "Value",
         "cheap vs sector on PE/PB with healthy returns - needs Milestone 4",
         False),
    Scan("oversold_quality", "Oversold + strong fundamentals",
         "technically oversold with solid fundamentals - needs Milestone 4",
         False),
]

_SCAN_BY_KEY = {s.key: s for s in SCANS}


def run_scan(key: str, entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    """Run a prebuilt scan. Returns [] for scans pending a later milestone."""
    scan = _SCAN_BY_KEY.get(key.lower())
    if scan is None:
        raise ValueError(f"unknown scan {key!r}; available: {list(_SCAN_BY_KEY)}")
    if not scan.ready or scan.predicate is None:
        return []
    return [e for e in entries if scan.predicate(e)]


def budget_picks(
    entries: list[LeaderboardEntry],
    capital: float,
    *,
    risk_pct: float = 1.5,
    top_n: int = 5,
):
    """Pair the top picks with a budget-aware trade plan (entry/stop/size)."""
    from risk.risk import build_trade_plan

    picks = []
    for entry in entries[:top_n]:
        if entry.ltp <= 0:
            continue
        plan = build_trade_plan(
            entry.symbol, entry.ltp, entry.atr, capital, risk_pct=risk_pct
        )
        picks.append((entry, plan))
    return picks
