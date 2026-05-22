"""Chart pattern detection: pivots, support/resistance, classic reversal
and continuation shapes.

These are deliberately heuristic - swing-pivot based - so they stay
explainable. The technical engine treats them as supporting evidence,
never as standalone triggers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Level:
    """A horizontal support or resistance level."""

    price: float
    kind: str        # support | resistance
    touches: int     # how many pivots cluster here
    distance_pct: float  # signed % from the latest close (+ above, - below)


@dataclass(frozen=True)
class ChartPattern:
    name: str
    bias: str        # bullish | bearish | neutral
    detail: str


# --------------------------------------------------------------------------
# swing pivots
# --------------------------------------------------------------------------
def _pivots(series: pd.Series, left: int, right: int, kind: str) -> list[int]:
    """Return integer positions of swing pivots (kind = 'high' | 'low')."""
    vals = series.to_numpy()
    n = len(vals)
    out: list[int] = []
    for i in range(left, n - right):
        window = vals[i - left : i + right + 1]
        pivot = vals[i]
        if kind == "high" and pivot == window.max() and (window == pivot).sum() == 1:
            out.append(i)
        elif kind == "low" and pivot == window.min() and (window == pivot).sum() == 1:
            out.append(i)
    return out


def pivot_highs(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[int]:
    return _pivots(df["high"], left, right, "high")


def pivot_lows(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[int]:
    return _pivots(df["low"], left, right, "low")


# --------------------------------------------------------------------------
# support / resistance
# --------------------------------------------------------------------------
def support_resistance(
    df: pd.DataFrame, left: int = 3, right: int = 3, tol_pct: float = 1.0
) -> list[Level]:
    """Cluster swing pivots into horizontal S/R levels, nearest first."""
    if len(df) < left + right + 2:
        return []

    close = float(df["close"].iloc[-1])
    raw: list[tuple[float, str]] = []
    for i in pivot_highs(df, left, right):
        raw.append((float(df["high"].iloc[i]), "resistance"))
    for i in pivot_lows(df, left, right):
        raw.append((float(df["low"].iloc[i]), "support"))

    levels: list[Level] = []
    used = [False] * len(raw)
    for i, (price, kind) in enumerate(raw):
        if used[i]:
            continue
        cluster = [price]
        used[i] = True
        for j in range(i + 1, len(raw)):
            if used[j] or raw[j][1] != kind:
                continue
            if abs(raw[j][0] - price) / price * 100.0 <= tol_pct:
                cluster.append(raw[j][0])
                used[j] = True
        lvl_price = float(np.mean(cluster))
        levels.append(
            Level(
                price=round(lvl_price, 2),
                kind=kind,
                touches=len(cluster),
                distance_pct=round((lvl_price - close) / close * 100.0, 2),
            )
        )
    return sorted(levels, key=lambda lv: abs(lv.distance_pct))


def nearest_support(levels: list[Level]) -> Level | None:
    below = [lv for lv in levels if lv.kind == "support" and lv.distance_pct < 0]
    return max(below, key=lambda lv: lv.distance_pct) if below else None


def nearest_resistance(levels: list[Level]) -> Level | None:
    above = [lv for lv in levels if lv.kind == "resistance" and lv.distance_pct > 0]
    return min(above, key=lambda lv: lv.distance_pct) if above else None


# --------------------------------------------------------------------------
# reversal / continuation shapes
# --------------------------------------------------------------------------
def _close(df: pd.DataFrame, prices: list[int], series: str) -> list[float]:
    return [float(df[series].iloc[i]) for i in prices]


def detect_double_top(df: pd.DataFrame, tol_pct: float = 3.0) -> ChartPattern | None:
    highs = pivot_highs(df)
    if len(highs) < 2:
        return None
    a, b = highs[-2], highs[-1]
    pa, pb = float(df["high"].iloc[a]), float(df["high"].iloc[b])
    if abs(pa - pb) / max(pa, pb) * 100.0 > tol_pct:
        return None
    trough = df["low"].iloc[a:b].min()
    if trough >= min(pa, pb):
        return None
    return ChartPattern(
        "double_top", "bearish", f"twin peaks near {round((pa + pb) / 2, 2)}"
    )


def detect_double_bottom(df: pd.DataFrame, tol_pct: float = 3.0) -> ChartPattern | None:
    lows = pivot_lows(df)
    if len(lows) < 2:
        return None
    a, b = lows[-2], lows[-1]
    pa, pb = float(df["low"].iloc[a]), float(df["low"].iloc[b])
    if abs(pa - pb) / max(pa, pb) * 100.0 > tol_pct:
        return None
    peak = df["high"].iloc[a:b].max()
    if peak <= max(pa, pb):
        return None
    return ChartPattern(
        "double_bottom", "bullish", f"twin troughs near {round((pa + pb) / 2, 2)}"
    )


def detect_head_and_shoulders(
    df: pd.DataFrame, shoulder_tol_pct: float = 5.0
) -> ChartPattern | None:
    highs = pivot_highs(df)
    if len(highs) >= 3:
        l, h, r = _close(df, highs[-3:], "high")
        if h > l and h > r and abs(l - r) / max(l, r) * 100.0 <= shoulder_tol_pct:
            return ChartPattern(
                "head_and_shoulders", "bearish", "topping reversal structure"
            )
    return None


def detect_inverse_head_and_shoulders(
    df: pd.DataFrame, shoulder_tol_pct: float = 5.0
) -> ChartPattern | None:
    lows = pivot_lows(df)
    if len(lows) >= 3:
        l, h, r = _close(df, lows[-3:], "low")
        if h < l and h < r and abs(l - r) / max(l, r) * 100.0 <= shoulder_tol_pct:
            return ChartPattern(
                "inverse_head_and_shoulders", "bullish", "bottoming reversal structure"
            )
    return None


def detect_rounding(df: pd.DataFrame, window: int = 20) -> ChartPattern | None:
    """Rounded top / bottom via the curvature of a quadratic fit (experimental)."""
    if len(df) < window:
        return None
    y = df["close"].iloc[-window:].to_numpy()
    x = np.arange(window)
    curvature = np.polyfit(x, y, 2)[0]
    span = y.max() - y.min()
    if span <= 0:
        return None
    norm = curvature / span
    if norm > 0.002:
        return ChartPattern("rounding_bottom", "bullish", "saucer base forming")
    if norm < -0.002:
        return ChartPattern("rounding_top", "bearish", "distribution dome forming")
    return None


def trend_channel(df: pd.DataFrame, window: int = 30) -> ChartPattern | None:
    """Classify the recent slope of price as a rising / falling channel."""
    if len(df) < window:
        return None
    y = df["close"].iloc[-window:].to_numpy()
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    avg = float(np.mean(y))
    slope_pct = slope / avg * 100.0
    if slope_pct > 0.15:
        return ChartPattern("rising_channel", "bullish", "price riding higher")
    if slope_pct < -0.15:
        return ChartPattern("falling_channel", "bearish", "price stair-stepping down")
    return ChartPattern("sideways_channel", "neutral", "rangebound")


def detect_patterns(df: pd.DataFrame) -> list[ChartPattern]:
    """Run every chart-pattern detector; return the ones that fired."""
    found: list[ChartPattern] = []
    for fn in (
        detect_double_top,
        detect_double_bottom,
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
        detect_rounding,
        trend_channel,
    ):
        pat = fn(df)
        if pat is not None:
            found.append(pat)
    return found
