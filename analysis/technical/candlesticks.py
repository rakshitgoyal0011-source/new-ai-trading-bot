"""Candlestick pattern recognition.

Every detector is vectorised and returns a boolean Series aligned to the
input frame (True on the bar where the pattern *completes*). Single-bar
reversal patterns (hammer / hanging man, shooting star / inverted hammer)
share a geometry but differ by the prior trend, so the prior trend is
folded into the detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class PatternHit:
    name: str
    bias: str        # bullish | bearish | neutral
    bars_ago: int    # 0 = most recent bar


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _geometry(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l)
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    return {
        "o": o, "h": h, "l": l, "c": c,
        "body": body,
        "range": rng,
        "upper": h - body_top,
        "lower": body_bot - l,
        "bull": c > o,
        "bear": c < o,
    }


def _prior_trend(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """+1 up / -1 down / 0 flat, measured over the bars *before* the signal."""
    ref = df["close"].shift(1)
    past = df["close"].shift(1 + lookback)
    return pd.Series(
        np.where(ref > past * 1.01, 1, np.where(ref < past * 0.99, -1, 0)),
        index=df.index,
    )


# --------------------------------------------------------------------------
# single-bar patterns
# --------------------------------------------------------------------------
def doji(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    return (g["range"] > 0) & (g["body"] <= 0.1 * g["range"])


def _hammer_shape(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    return (
        (g["range"] > 0)
        & (g["body"] > 0)
        & (g["lower"] >= 2.0 * g["body"])
        & (g["upper"] <= g["body"])
        & (g["body"] <= 0.4 * g["range"])
    )


def _star_shape(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    return (
        (g["range"] > 0)
        & (g["body"] > 0)
        & (g["upper"] >= 2.0 * g["body"])
        & (g["lower"] <= g["body"])
        & (g["body"] <= 0.4 * g["range"])
    )


def hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish reversal: hammer shape after a downtrend."""
    return _hammer_shape(df) & (_prior_trend(df) < 0)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """Bearish reversal: hammer shape after an uptrend."""
    return _hammer_shape(df) & (_prior_trend(df) > 0)


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish reversal: star shape after a downtrend."""
    return _star_shape(df) & (_prior_trend(df) < 0)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Bearish reversal: star shape after an uptrend."""
    return _star_shape(df) & (_prior_trend(df) > 0)


# --------------------------------------------------------------------------
# two-bar patterns
# --------------------------------------------------------------------------
def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    prev_bear = g["bear"].shift(1).fillna(False)
    return (
        prev_bear
        & g["bull"]
        & (df["close"] >= df["open"].shift(1))
        & (df["open"] <= df["close"].shift(1))
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    prev_bull = g["bull"].shift(1).fillna(False)
    return (
        prev_bull
        & g["bear"]
        & (df["open"] >= df["close"].shift(1))
        & (df["close"] <= df["open"].shift(1))
    )


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    prev_bear = g["bear"].shift(1).fillna(False)
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    return (
        prev_bear
        & g["bull"]
        & (body_top <= df["open"].shift(1))
        & (body_bot >= df["close"].shift(1))
        & (g["body"].shift(1) > g["body"])
    )


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    prev_bull = g["bull"].shift(1).fillna(False)
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    return (
        prev_bull
        & g["bear"]
        & (body_top <= df["close"].shift(1))
        & (body_bot >= df["open"].shift(1))
        & (g["body"].shift(1) > g["body"])
    )


# --------------------------------------------------------------------------
# three-bar patterns
# --------------------------------------------------------------------------
def morning_star(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    c1_bear = g["bear"].shift(2).fillna(False)
    small_mid = g["body"].shift(1) <= 0.5 * g["body"].shift(2)
    c3_bull = g["bull"]
    c1_mid = (df["open"].shift(2) + df["close"].shift(2)) / 2.0
    return c1_bear & small_mid & c3_bull & (df["close"] > c1_mid)


def evening_star(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    c1_bull = g["bull"].shift(2).fillna(False)
    small_mid = g["body"].shift(1) <= 0.5 * g["body"].shift(2)
    c3_bear = g["bear"]
    c1_mid = (df["open"].shift(2) + df["close"].shift(2)) / 2.0
    return c1_bull & small_mid & c3_bear & (df["close"] < c1_mid)


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    bull = g["bull"]
    three_bull = bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False)
    rising = (df["close"] > df["close"].shift(1)) & (
        df["close"].shift(1) > df["close"].shift(2)
    )
    open_in_prev = (df["open"] <= df["close"].shift(1)) & (
        df["open"] >= df["open"].shift(1)
    )
    return three_bull & rising & open_in_prev


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    g = _geometry(df)
    bear = g["bear"]
    three_bear = bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False)
    falling = (df["close"] < df["close"].shift(1)) & (
        df["close"].shift(1) < df["close"].shift(2)
    )
    open_in_prev = (df["open"] >= df["close"].shift(1)) & (
        df["open"] <= df["open"].shift(1)
    )
    return three_bear & falling & open_in_prev


# --------------------------------------------------------------------------
# registry + convenience
# --------------------------------------------------------------------------
PATTERNS: dict[str, tuple] = {
    "hammer": (hammer, BULLISH),
    "hanging_man": (hanging_man, BEARISH),
    "inverted_hammer": (inverted_hammer, BULLISH),
    "shooting_star": (shooting_star, BEARISH),
    "doji": (doji, NEUTRAL),
    "bullish_engulfing": (bullish_engulfing, BULLISH),
    "bearish_engulfing": (bearish_engulfing, BEARISH),
    "bullish_harami": (bullish_harami, BULLISH),
    "bearish_harami": (bearish_harami, BEARISH),
    "morning_star": (morning_star, BULLISH),
    "evening_star": (evening_star, BEARISH),
    "three_white_soldiers": (three_white_soldiers, BULLISH),
    "three_black_crows": (three_black_crows, BEARISH),
}


def detect_all(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Run every detector; return name -> boolean Series."""
    return {name: func(df) for name, (func, _) in PATTERNS.items()}


def recent_hits(df: pd.DataFrame, lookback: int = 3) -> list[PatternHit]:
    """Patterns that completed within the last `lookback` bars (newest first)."""
    if len(df) < 3:
        return []
    hits: list[PatternHit] = []
    window = min(lookback, len(df))
    for name, (func, bias) in PATTERNS.items():
        series = func(df).fillna(False)
        for bars_ago in range(window):
            if bool(series.iloc[-1 - bars_ago]):
                hits.append(PatternHit(name, bias, bars_ago))
    return sorted(hits, key=lambda h: h.bars_ago)
