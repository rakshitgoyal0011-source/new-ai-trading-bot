"""Tests for the backtest cost model (Indian-market charges)."""
from backtest.backtest import CostModel


def test_round_trip_cost_is_positive():
    assert CostModel().round_trip(100_000.0, 101_000.0) > 0


def test_cost_scales_with_turnover():
    cm = CostModel()
    small = cm.round_trip(10_000.0, 10_000.0)
    big = cm.round_trip(100_000.0, 100_000.0)
    assert big > small


def test_brokerage_is_capped_per_leg():
    # at a huge notional the % brokerage is capped, so it cannot dominate
    cm = CostModel(slippage_bps=0.0)
    cost = cm.round_trip(1e7, 1e7)
    # both legs capped at brokerage_cap -> brokerage portion <= 2 * cap
    assert cost > 0
    assert cm._brokerage(1e7) == cm.brokerage_cap


def test_zero_slippage_is_cheaper():
    base = CostModel()
    no_slip = CostModel(slippage_bps=0.0)
    assert no_slip.round_trip(100_000.0, 100_000.0) < base.round_trip(
        100_000.0, 100_000.0)
