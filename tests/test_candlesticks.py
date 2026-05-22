"""Tests for candlestick pattern detection using hand-crafted candles."""
import pandas as pd

from analysis.technical import candlesticks as cs


def _df(rows):
    """rows = list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def test_doji_detected():
    df = _df([(100, 101, 99, 100), (100, 105, 95, 100.05)])
    assert bool(cs.doji(df).iloc[-1])


def test_bullish_engulfing():
    # prev bar bearish, last bar bullish and engulfs it
    df = _df([(110, 111, 104, 105), (105, 106, 99, 100), (99, 107, 98, 106)])
    assert bool(cs.bullish_engulfing(df).iloc[-1])


def test_bearish_engulfing():
    df = _df([(100, 106, 99, 100), (100, 107, 99, 105), (106, 107, 99, 100)])
    assert bool(cs.bearish_engulfing(df).iloc[-1])


def test_hammer_in_downtrend():
    # 7 declining bars to establish a downtrend, then a hammer
    df = _df([
        (120, 121, 119, 119), (118, 119, 116, 116), (115, 116, 113, 113),
        (112, 113, 110, 110), (109, 110, 107, 107), (106, 107, 104, 104),
        (103, 104, 101, 101),
        (100.0, 100.05, 94.0, 99.7),  # small body on top, long lower shadow
    ])
    assert bool(cs.hammer(df).iloc[-1])


def test_morning_star():
    df = _df([
        (100, 101, 99, 100), (101, 102, 100, 101),         # padding
        (110, 111, 99, 100),                                # big bearish
        (99, 101, 98, 99.5),                                # small body
        (100, 112, 99, 110),                                # big bullish
    ])
    assert bool(cs.morning_star(df).iloc[-1])


def test_three_white_soldiers():
    df = _df([
        (100, 101, 99, 100), (100, 101, 99, 100),           # padding
        (100, 105, 99, 104),
        (102, 108, 101, 107),
        (105, 111, 104, 110),
    ])
    assert bool(cs.three_white_soldiers(df).iloc[-1])


def test_recent_hits_returns_pattern():
    df = _df([(110, 111, 104, 105), (105, 106, 99, 100), (99, 107, 98, 106)])
    hits = cs.recent_hits(df, lookback=1)
    assert any(h.name == "bullish_engulfing" and h.bias == "bullish" for h in hits)
