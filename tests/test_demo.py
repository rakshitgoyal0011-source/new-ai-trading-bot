"""Tests for the synthetic demo market."""
from data.demo import DemoMarket


def test_history_is_deterministic_for_a_seed():
    a = DemoMarket(seed=1).history("RELIANCE", bars=100)
    b = DemoMarket(seed=1).history("RELIANCE", bars=100)
    assert len(a) == 100
    assert a["close"].iloc[-1] == b["close"].iloc[-1]


def test_history_ohlc_is_internally_consistent():
    df = DemoMarket().history("TCS", bars=150)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["volume"] > 0).all()


def test_quote_advances_on_step():
    market = DemoMarket()
    first = market.quote("INFY").ltp
    for _ in range(20):
        market.step(["INFY"])
    moved = market.quote("INFY").ltp
    assert first > 0 and moved > 0
    assert first != moved  # random walk should have moved the price
