"""Tests for the composite buy-probability engine."""
from analysis.composite.engine import CompositeEngine
from config.weights import SignalWeights


def test_technical_only_blend_equals_technical():
    res = CompositeEngine().combine("X", technical_score=80.0)
    assert res.composite_score == 80.0
    assert res.bias == "bullish"
    assert res.weights_used == {"technical": 1.0}


def test_all_three_engine_blend():
    eng = CompositeEngine(SignalWeights(0.5, 0.3, 0.2))
    res = eng.combine("X", 80.0, 60.0, 40.0)
    # 0.5*80 + 0.3*60 + 0.2*40 = 66
    assert res.composite_score == 66.0
    assert set(res.components) == {"technical", "fundamental", "news"}


def test_missing_news_renormalises_weights():
    eng = CompositeEngine(SignalWeights(0.45, 0.35, 0.20))
    res = eng.combine("X", 80.0, 60.0)  # fundamental present, news absent
    expected = (80.0 * 0.45 + 60.0 * 0.35) / 0.80
    assert abs(res.composite_score - round(expected, 1)) < 0.1
    assert "news" not in res.weights_used


def test_probability_is_not_faked_pre_calibration():
    res = CompositeEngine().combine("X", 90.0)
    # an uncalibrated composite must NOT expose a probability
    assert res.calibrated_probability is None
    assert res.disclaimer and "NOT FINANCIAL ADVICE" in res.disclaimer


def test_low_score_is_bearish():
    res = CompositeEngine().combine("X", technical_score=25.0)
    assert res.bias == "bearish"
