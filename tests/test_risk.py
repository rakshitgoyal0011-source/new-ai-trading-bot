"""Tests for the risk & discipline layer."""
from risk.risk import (
    atr_stop,
    build_trade_plan,
    position_size,
    reward_risk,
    risk_rules,
)


def test_position_size_known_value():
    # capital 100000, 2% risk = 2000 budget, 10/share risk -> 200 shares
    assert position_size(100_000, 2.0, 10.0) == 200


def test_position_size_zero_risk_is_safe():
    assert position_size(100_000, 2.0, 0.0) == 0


def test_atr_stop_long():
    assert atr_stop(100.0, 2.0, 2.0, "long") == 96.0


def test_atr_stop_short():
    assert atr_stop(100.0, 2.0, 2.0, "short") == 104.0


def test_reward_risk_ratio():
    assert reward_risk(100.0, 95.0, 110.0, "long") == 2.0


def test_trade_plan_basic_math():
    plan = build_trade_plan("TEST", 100.0, 2.0, 100_000,
                            risk_pct=2.0, atr_multiplier=2.0)
    assert plan.stop == 96.0
    assert plan.risk_per_share == 4.0
    assert plan.shares == 500            # 2000 budget / 4 per share
    assert plan.targets[0] == 104.0      # entry + 1R
    assert plan.reward_risk == 2.0
    assert plan.position_value == 50_000.0


def test_trade_plan_capped_by_budget():
    # pricey stock, capital cannot afford the risk-sized position
    plan = build_trade_plan("PRICEY", 5000.0, 50.0, 100_000, risk_pct=2.0)
    assert plan.shares <= 20
    assert plan.position_value <= 100_000.0


def test_trade_plan_handles_zero_atr():
    plan = build_trade_plan("TEST", 100.0, 0.0, 100_000)
    assert plan.stop < 100.0             # fell back to a % stop
    assert any("ATR" in n for n in plan.notes)


def test_risk_rules_present():
    rules = risk_rules()
    assert len(rules) >= 8
    assert all(isinstance(r, str) and r for r in rules)
