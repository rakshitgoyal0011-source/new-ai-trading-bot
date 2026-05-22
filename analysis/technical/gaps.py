"""Gap theory: detect price gaps and classify them.

A gap is unfilled white space between one bar's range and the next.
Classification (common / breakaway / runaway / exhaustion / island) is
heuristic - it leans on prior trend, the size of the extended move and
the volume surge - and the engine weights it accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

COMMON = "common"
BREAKAWAY = "breakaway"
RUNAWAY = "runaway"
EXHAUSTION = "exhaustion"
ISLAND = "island"


@dataclass(frozen=True)
class Gap:
    bars_ago: int
    direction: str       # up | down
    size_pct: float      # gap size as % of prior close
    kind: str            # common | breakaway | runaway | exhaustion | island
    volume_surge: float  # volume vs 20-bar average on the gap bar
    filled: bool         # has price since traded back through the gap

    @property
    def bias(self) -> str:
        """Directional read of the gap for the composite signal."""
        if self.kind == EXHAUSTION:
            # exhaustion warns of a reversal *against* the gap direction
            return "bearish" if self.direction == "up" else "bullish"
        if self.kind == COMMON:
            return "neutral"
        return "bullish" if self.direction == "up" else "bearish"


def detect_gaps(df: pd.DataFrame, lookback: int = 40) -> list[Gap]:
    """Return classified gaps within the last `lookback` bars, newest first."""
    n = len(df)
    if n < 25:
        return []

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()

    start = max(22, n - lookback)
    raw: list[dict] = []
    for i in range(start, n):
        if low[i] > high[i - 1]:
            direction, size = "up", (low[i] - high[i - 1]) / close[i - 1]
        elif high[i] < low[i - 1]:
            direction, size = "down", (low[i - 1] - high[i]) / close[i - 1]
        else:
            continue
        raw.append({"i": i, "direction": direction, "size_pct": size * 100.0})

    gaps: list[Gap] = []
    for g in raw:
        i, direction = g["i"], g["direction"]
        size_pct = g["size_pct"]

        avg_vol = vol[i - 20 : i].mean() or 1.0
        vol_surge = float(vol[i] / avg_vol)
        move20 = (close[i - 1] - close[i - 21]) / close[i - 21] * 100.0
        trend_up = move20 > 4.0
        trend_dn = move20 < -4.0
        extended = abs(move20) > 15.0

        # filled?
        if direction == "up":
            filled = bool((low[i + 1 :] <= high[i - 1]).any()) if i + 1 < n else False
        else:
            filled = bool((high[i + 1 :] >= low[i - 1]).any()) if i + 1 < n else False

        # island: an opposite-direction gap within 6 bars
        island = any(
            o["direction"] != direction and 0 < o["i"] - i <= 6
            for o in raw
        ) or any(
            o["direction"] != direction and 0 < i - o["i"] <= 6
            for o in raw
        )

        if island:
            kind = ISLAND
        elif size_pct < 0.4:
            kind = COMMON
        elif (direction == "up" and trend_up) or (direction == "down" and trend_dn):
            kind = EXHAUSTION if (extended and vol_surge > 2.0) else RUNAWAY
        elif not trend_up and not trend_dn:
            kind = BREAKAWAY
        else:
            # gap against the prevailing trend - a reversal breakaway
            kind = BREAKAWAY

        gaps.append(
            Gap(
                bars_ago=n - 1 - i,
                direction=direction,
                size_pct=round(size_pct, 2),
                kind=kind,
                volume_surge=round(vol_surge, 2),
                filled=filled,
            )
        )

    return sorted(gaps, key=lambda x: x.bars_ago)
