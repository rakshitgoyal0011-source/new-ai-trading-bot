"""Tests for the rolling OHLC candle builder."""
from datetime import datetime, timedelta

import pytest

from data.candles import CandleBuilder


def test_ticks_fold_into_one_candle():
    cb = CandleBuilder("X", "1m")
    t = datetime(2024, 1, 1, 9, 15, 0)
    cb.add_tick(100.0, 10.0, t)
    cb.add_tick(102.0, 5.0, t + timedelta(seconds=10))
    cb.add_tick(99.0, 5.0, t + timedelta(seconds=20))
    row = cb.to_dataframe().iloc[-1]
    assert row.open == 100.0 and row.high == 102.0
    assert row.low == 99.0 and row.close == 99.0
    assert row.volume == 20.0


def test_candle_rolls_over_on_new_bar():
    cb = CandleBuilder("X", "1m")
    t = datetime(2024, 1, 1, 9, 15, 0)
    cb.add_tick(100.0, 1.0, t)
    cb.add_tick(105.0, 1.0, t + timedelta(minutes=1))
    assert cb.bar_count == 2
    df = cb.to_dataframe()
    assert df.iloc[0].close == 100.0
    assert df.iloc[1].open == 105.0


def test_rolling_window_is_bounded():
    cb = CandleBuilder("X", "1m", max_bars=5)
    base = datetime(2024, 1, 1, 9, 15, 0)
    for i in range(20):
        cb.add_tick(100.0 + i, 1.0, base + timedelta(minutes=i))
    assert cb.bar_count <= 6  # max_bars completed + 1 in-progress


def test_unknown_interval_rejected():
    with pytest.raises(ValueError):
        CandleBuilder("X", "7m")
