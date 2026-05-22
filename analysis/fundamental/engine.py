"""Fundamental analysis engine.

Sector-relative scoring: every metric a stock is graded on is compared
to its sector peers (lower PE / PB / debt is better; higher ROE /
margins / growth is better). Five sub-scores - valuation, profitability,
growth, financial health, quality flags - blend into a 0-100
FUNDAMENTAL SCORE with reasons that name the peer comparison.

Why sector-relative: 'good PE' for a Bank looks nothing like 'good PE'
for an IT services company. Absolute cutoffs lie.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from data.fundamentals import FundamentalData


@dataclass
class FundamentalResult:
    symbol: str
    score: float                              # 0-100
    bias: str
    available: bool = False
    reasons: list[str] = field(default_factory=list)
    sub_scores: dict[str, float] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    peers_compared: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# (attribute, label, higher_is_better, sub-score family)
_METRICS: list[tuple[str, str, bool, str]] = [
    ("pe",                  "PE",                 False, "valuation"),
    ("pb",                  "PB",                 False, "valuation"),
    ("peg",                 "PEG",                False, "valuation"),
    ("roe",                 "ROE %",              True,  "profitability"),
    ("roce",                "ROCE %",             True,  "profitability"),
    ("operating_margin",    "Operating margin %", True,  "profitability"),
    ("net_margin",          "Net margin %",       True,  "profitability"),
    ("revenue_growth",      "Revenue growth %",   True,  "growth"),
    ("eps_growth",          "EPS growth %",       True,  "growth"),
    ("debt_to_equity",      "Debt / Equity",      False, "financial_health"),
    ("interest_coverage",   "Interest coverage",  True,  "financial_health"),
]


def _percentile(target: float | None, peer_vals: list[float], higher_better: bool) -> float | None:
    """Return target's percentile among peers (0 worst .. 100 best)."""
    peer_vals = [v for v in peer_vals if v is not None]
    if target is None or not peer_vals:
        return None
    if higher_better:
        beats = sum(1 for v in peer_vals if target > v)
    else:
        beats = sum(1 for v in peer_vals if target < v)
    return 100.0 * beats / len(peer_vals)


class FundamentalEngine:
    """Sector-relative fundamental scorer."""

    SUB_SCORES = ("valuation", "profitability", "growth",
                  "financial_health", "quality")

    def analyze(
        self,
        symbol: str,
        data: FundamentalData | None,
        peers: list[FundamentalData] | None = None,
    ) -> FundamentalResult:
        if data is None or not data.available:
            return FundamentalResult(
                symbol=symbol, score=50.0, bias="neutral", available=False,
                reasons=["fundamental data unavailable for this symbol"],
                peers_compared=0,
            )

        peers = [p for p in (peers or []) if p is not None and p.available]
        per_family: dict[str, list[float]] = {f: [] for f in self.SUB_SCORES[:-1]}
        reasons: list[str] = []

        for attr, label, higher_better, family in _METRICS:
            target = getattr(data, attr)
            peer_vals = [getattr(p, attr) for p in peers]
            pct = _percentile(target, peer_vals, higher_better)
            if pct is None:
                continue
            per_family[family].append(pct)
            if target is not None:
                reasons.append(
                    f"{label} {target:.2f} - better than {pct:.0f}% of peers"
                )

        sub_scores: dict[str, float] = {
            f: round(sum(vs) / len(vs), 1) for f, vs in per_family.items() if vs
        }

        flags: list[str] = []
        if data.promoter_holding is not None and data.promoter_holding >= 50:
            flags.append("strong promoter holding")
        if data.promoter_pledge is not None and data.promoter_pledge >= 25:
            flags.append("promoter pledge above 25% - quality flag")
        if (data.revenue_growth or 0) > 0 and (data.roe or 0) > 12:
            flags.append("growing and profitable")
        if (data.debt_to_equity or 0) > 2 and (data.interest_coverage or 100) < 3:
            flags.append("highly levered with weak coverage")

        # quality sub-score: positive flags push up, negative push down
        positive = sum(
            f in ("strong promoter holding", "growing and profitable") for f in flags
        )
        negative = sum(
            f.startswith("promoter pledge") or f.startswith("highly levered")
            for f in flags
        )
        quality = 50.0 + 12.0 * (positive - negative)
        quality = max(0.0, min(100.0, quality))
        sub_scores["quality"] = round(quality, 1)

        score = (
            round(sum(sub_scores.values()) / len(sub_scores), 1)
            if sub_scores else 50.0
        )
        bias = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"

        # surface the most striking peer comparisons + the quality flags
        reasons.sort(
            key=lambda r: abs(float(r.split("better than ")[1].split("%")[0]) - 50.0),
            reverse=True,
        )
        if flags:
            reasons = flags + reasons[:6]
        else:
            reasons = reasons[:6]

        return FundamentalResult(
            symbol=symbol, score=score, bias=bias, available=True,
            reasons=reasons, sub_scores=sub_scores, quality_flags=flags,
            peers_compared=len(peers),
        )
