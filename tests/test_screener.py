"""Tests for the screener / TOP leaderboard ranking."""
from screener.screener import LeaderboardEntry, budget_picks, rank, run_scan


def _entries():
    return [
        LeaderboardEntry("A", sector="IT", composite_score=80.0,
                         technical_score=80.0, trend="uptrend", ltp=100.0,
                         atr=2.0, metrics={"rsi": 60.0, "volume_ratio": 2.5}),
        LeaderboardEntry("B", sector="Bank", composite_score=60.0,
                         technical_score=60.0, trend="sideways", ltp=200.0,
                         atr=5.0, metrics={"rsi": 50.0, "volume_ratio": 1.0}),
        LeaderboardEntry("C", sector="IT", composite_score=40.0,
                         technical_score=40.0, trend="downtrend", ltp=50.0,
                         atr=1.0, metrics={"rsi": 30.0, "volume_ratio": 1.2}),
    ]


def test_rank_orders_by_composite_desc():
    assert [e.symbol for e in rank(_entries())] == ["A", "B", "C"]


def test_rank_sector_filter():
    assert {e.symbol for e in rank(_entries(), sector="IT")} == {"A", "C"}


def test_rank_price_filter():
    assert {e.symbol for e in rank(_entries(), max_price=150.0)} == {"A", "C"}


def test_rank_max_results():
    assert len(rank(_entries(), max_results=1)) == 1


def test_scan_momentum_matches_strong_uptrend():
    hits = run_scan("momentum", _entries())
    assert [e.symbol for e in hits] == ["A"]


def test_scan_volume_spike():
    hits = run_scan("volume_spike", _entries())
    assert [e.symbol for e in hits] == ["A"]


def test_pending_scan_returns_empty():
    assert run_scan("value", _entries()) == []


def test_budget_picks_produce_trade_plans():
    picks = budget_picks(rank(_entries()), 100_000.0, top_n=2)
    assert len(picks) == 2
    for entry, plan in picks:
        assert plan.symbol == entry.symbol
        assert plan.shares >= 0
        assert plan.position_value <= 100_000.0
