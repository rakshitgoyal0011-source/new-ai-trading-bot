"""Risk & discipline layer.

Turns a suggested entry into a concrete, budget-aware trade plan:
ATR-based stop-loss, laddered targets, reward/risk ratio, position size
for the entered capital and trailing-stop guidance. Mirrors the risk
discipline taught in the NSE technical-analysis module (Ch. 7).

This module sizes and frames trades. It does NOT place orders.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class TradePlan:
    symbol: str
    side: str                       # long | short
    entry: float
    stop: float
    targets: list[float] = field(default_factory=list)
    reward_risk: float = 0.0
    shares: int = 0
    position_value: float = 0.0
    capital_at_risk: float = 0.0
    risk_pct: float = 0.0
    risk_per_share: float = 0.0
    trailing: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def position_size(capital: float, risk_pct: float, risk_per_share: float) -> int:
    """Shares to buy so that a stop-out loses exactly `risk_pct` of capital."""
    if risk_per_share <= 0 or capital <= 0:
        return 0
    risk_budget = capital * (risk_pct / 100.0)
    return int(math.floor(risk_budget / risk_per_share))


def atr_stop(entry: float, atr: float, multiplier: float, side: str) -> float:
    """Volatility-based stop: `multiplier` ATRs away from entry."""
    if side == "long":
        return round(entry - multiplier * atr, 2)
    return round(entry + multiplier * atr, 2)


def reward_risk(entry: float, stop: float, target: float, side: str = "long") -> float:
    """Reward-to-risk ratio for a single target."""
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = (target - entry) if side == "long" else (entry - target)
    return round(reward / risk, 2)


def build_trade_plan(
    symbol: str,
    entry: float,
    atr: float,
    capital: float,
    *,
    risk_pct: float = 1.5,
    side: str = "long",
    atr_multiplier: float = 2.0,
    target_multiples: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> TradePlan:
    """Construct a full, budget-aware trade plan from an entry + ATR.

    `target_multiples` are reward/risk multiples - e.g. 2.0 puts a target
    two units of risk away from entry.
    """
    notes: list[str] = []
    if atr <= 0:
        notes.append("ATR unavailable - stop falls back to 3% of entry")
        atr = entry * 0.03 / atr_multiplier

    stop = atr_stop(entry, atr, atr_multiplier, side)
    risk_per_share = abs(entry - stop)

    targets = [
        round(entry + m * risk_per_share, 2) if side == "long"
        else round(entry - m * risk_per_share, 2)
        for m in target_multiples
    ]

    shares = position_size(capital, risk_pct, risk_per_share)

    # never let the notional exceed the entered capital
    max_affordable = int(math.floor(capital / entry)) if entry > 0 else 0
    if shares > max_affordable:
        notes.append(
            f"risk-based size ({shares}) exceeds budget - capped to "
            f"{max_affordable} affordable shares"
        )
        shares = max_affordable
    if shares == 0:
        notes.append("capital too small for one risk-sized share of this stock")

    position_value = round(shares * entry, 2)
    capital_at_risk = round(shares * risk_per_share, 2)
    primary_rr = reward_risk(entry, stop, targets[1] if len(targets) > 1
                             else targets[0], side)
    if primary_rr < 1.5:
        notes.append(f"reward/risk {primary_rr} is below the 1.5 minimum - skip or re-frame")

    trailing = (
        f"once price moves +1R ({round(risk_per_share, 2)}/sh) in favour, "
        f"trail the stop by 1x ATR ({round(atr, 2)}) to lock gains"
    )

    return TradePlan(
        symbol=symbol, side=side, entry=round(entry, 2), stop=stop,
        targets=targets, reward_risk=primary_rr, shares=shares,
        position_value=position_value, capital_at_risk=capital_at_risk,
        risk_pct=round(risk_pct, 2), risk_per_share=round(risk_per_share, 2),
        trailing=trailing, notes=notes,
    )


RISK_RULES: list[str] = [
    "Risk only 1-2% of total capital on any single trade.",
    "Define the stop-loss BEFORE entering - never widen it afterwards.",
    "Size the position from the stop distance, not from conviction.",
    "Take trades only with a reward/risk ratio of at least 1.5:1.",
    "Trail the stop to break-even once price moves one unit of risk (1R) in favour.",
    "Never average down on a losing position to 'reduce' the cost.",
    "Cap total open risk (portfolio heat) at roughly 5-6% of capital.",
    "Trade with the dominant trend; counter-trend trades need smaller size.",
    "One setup = one plan: entry, stop, target and size written down first.",
    "A probability is an estimate, not a promise - survive the losers.",
]


def risk_rules() -> list[str]:
    """Return the risk-discipline checklist surfaced by the RULES command."""
    return list(RISK_RULES)
