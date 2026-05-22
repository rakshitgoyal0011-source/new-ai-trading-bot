"""Tests for the fundamentals provider + sector-relative engine."""
from analysis.fundamental.engine import FundamentalEngine, _percentile
from data.fundamentals import DemoFundamentals, FundamentalData


def test_percentile_higher_is_better():
    assert _percentile(100.0, [80.0, 90.0, 95.0, 110.0],
                       higher_better=True) == 75.0


def test_percentile_lower_is_better():
    assert _percentile(100.0, [80.0, 90.0, 95.0, 110.0],
                       higher_better=False) == 25.0


def test_percentile_handles_missing_data():
    assert _percentile(None, [1.0, 2.0], higher_better=True) is None
    assert _percentile(50.0, [], higher_better=True) is None


def test_demo_fundamentals_is_deterministic():
    a = DemoFundamentals().fetch("RELIANCE")
    b = DemoFundamentals().fetch("RELIANCE")
    assert a.pe == b.pe and a.roe == b.roe
    assert a.available is True


def test_engine_returns_neutral_when_data_unavailable():
    res = FundamentalEngine().analyze(
        "X", FundamentalData(symbol="X", available=False)
    )
    assert res.score == 50.0
    assert res.available is False


def test_engine_best_in_class_scores_high():
    target = FundamentalData(
        symbol="T", available=True,
        pe=10.0, pb=2.0, roe=25.0, debt_to_equity=0.3,
        revenue_growth=20.0, operating_margin=20.0, net_margin=15.0,
    )
    peers = [
        FundamentalData(symbol="A", available=True, pe=20.0, pb=3.0, roe=15.0,
                        debt_to_equity=0.6, revenue_growth=10.0,
                        operating_margin=12.0, net_margin=8.0),
        FundamentalData(symbol="B", available=True, pe=30.0, pb=5.0, roe=8.0,
                        debt_to_equity=1.0, revenue_growth=5.0,
                        operating_margin=9.0, net_margin=5.0),
        FundamentalData(symbol="C", available=True, pe=25.0, pb=4.0, roe=12.0,
                        debt_to_equity=0.8, revenue_growth=8.0,
                        operating_margin=11.0, net_margin=6.0),
    ]
    res = FundamentalEngine().analyze("T", target, peers)
    assert res.available is True
    assert res.score > 70.0
    assert res.bias == "bullish"
    assert res.peers_compared == 3
    assert res.sub_scores


def test_engine_worst_in_class_scores_low():
    target = FundamentalData(
        symbol="T", available=True,
        pe=50.0, pb=10.0, roe=2.0, debt_to_equity=2.5,
        revenue_growth=-5.0, operating_margin=2.0, net_margin=1.0,
    )
    peers = [
        FundamentalData(symbol="A", available=True, pe=20.0, pb=3.0, roe=15.0,
                        debt_to_equity=0.6, revenue_growth=10.0,
                        operating_margin=12.0, net_margin=8.0),
        FundamentalData(symbol="B", available=True, pe=15.0, pb=2.0, roe=20.0,
                        debt_to_equity=0.4, revenue_growth=15.0,
                        operating_margin=15.0, net_margin=10.0),
    ]
    res = FundamentalEngine().analyze("T", target, peers)
    assert res.score < 35.0
    assert res.bias == "bearish"
