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


def test_all_scans_are_now_ready_after_m6():
    from screener.screener import SCANS
    assert all(s.ready for s in SCANS)


def test_value_scan_uses_fundamental_metric():
    e1 = LeaderboardEntry(
        "A", sector="IT", composite_score=70.0, technical_score=70.0,
        ltp=100.0, atr=1.0,
        metrics={"fundamental_score": 75.0, "sentiment_score": 50.0,
                 "gap_pct": 0.0, "from_52w_high_pct": 50.0, "atr_pct": 1.0},
    )
    e2 = LeaderboardEntry(
        "B", sector="IT", composite_score=70.0, technical_score=70.0,
        ltp=100.0, atr=1.0,
        metrics={"fundamental_score": 50.0, "sentiment_score": 50.0,
                 "gap_pct": 0.0, "from_52w_high_pct": 50.0, "atr_pct": 1.0},
    )
    assert [e.symbol for e in run_scan("value", [e1, e2])] == ["A"]


def test_gap_up_scan():
    e1 = LeaderboardEntry(
        "A", composite_score=60.0, technical_score=70.0,
        metrics={"gap_pct": 3.5, "from_52w_high_pct": 50.0, "atr_pct": 1.0},
    )
    e2 = LeaderboardEntry(
        "B", composite_score=60.0, technical_score=70.0,
        metrics={"gap_pct": 1.0, "from_52w_high_pct": 50.0, "atr_pct": 1.0},
    )
    e3 = LeaderboardEntry(
        "C", composite_score=60.0, technical_score=40.0,
        metrics={"gap_pct": 3.5, "from_52w_high_pct": 50.0, "atr_pct": 1.0},
    )
    assert [e.symbol for e in run_scan("gap_up", [e1, e2, e3])] == ["A"]


def test_near_52w_high_scan():
    e1 = LeaderboardEntry(
        "A", bias="bullish", composite_score=70.0,
        metrics={"from_52w_high_pct": 97.0, "atr_pct": 1.0, "gap_pct": 0.0},
    )
    e2 = LeaderboardEntry(
        "B", bias="bullish", composite_score=70.0,
        metrics={"from_52w_high_pct": 80.0, "atr_pct": 1.0, "gap_pct": 0.0},
    )
    e3 = LeaderboardEntry(
        "C", bias="bearish", composite_score=70.0,
        metrics={"from_52w_high_pct": 99.0, "atr_pct": 1.0, "gap_pct": 0.0},
    )
    assert [e.symbol for e in run_scan("near_52w_high", [e1, e2, e3])] == ["A"]


def test_low_vol_uptrend_scan():
    e1 = LeaderboardEntry(
        "A", trend="uptrend", technical_score=70.0,
        metrics={"atr_pct": 1.5, "from_52w_high_pct": 50.0, "gap_pct": 0.0},
    )
    e2 = LeaderboardEntry(
        "B", trend="uptrend", technical_score=70.0,
        metrics={"atr_pct": 4.0, "from_52w_high_pct": 50.0, "gap_pct": 0.0},
    )
    e3 = LeaderboardEntry(
        "C", trend="sideways", technical_score=70.0,
        metrics={"atr_pct": 1.5, "from_52w_high_pct": 50.0, "gap_pct": 0.0},
    )
    out = run_scan("low_vol_uptrend", [e1, e2, e3])
    assert [e.symbol for e in out] == ["A"]


def test_budget_picks_produce_trade_plans():
    picks = budget_picks(rank(_entries()), 100_000.0, top_n=2)
    assert len(picks) == 2
    for entry, plan in picks:
        assert plan.symbol == entry.symbol
        assert plan.shares >= 0
        assert plan.position_value <= 100_000.0
