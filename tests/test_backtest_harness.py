"""End-to-end tests for the backtest harness + composite calibrator wiring."""
from __future__ import annotations

from analysis.composite.engine import CompositeEngine
from backtest.calibration import Calibrator
from backtest.events import generate_points
from backtest.harness import run_backtest
from config.settings import get_settings


def test_generate_points_produces_valid_pairs():
    points = generate_points(
        get_settings(),
        symbols=["RELIANCE", "TCS", "INFY"],
        horizon_bars=5,
        lookback_bars=200,
    )
    assert len(points) > 100
    for p in points[:50]:
        assert 0.0 <= p.composite_score <= 100.0
        assert p.label in (0, 1)
        # forward return / label invariant
        assert (p.forward_return > 0) == (p.label == 1)


def test_run_backtest_writes_a_calibrator(tmp_path):
    save = tmp_path / "cal.json"
    report = run_backtest(
        get_settings(),
        symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        horizon_bars=5,
        save_path=save,
    )
    assert save.exists()
    assert report.n_points > 100
    cal = Calibrator.load(save)
    # baseline brier should be positive (the base rate is not 0 or 1)
    assert cal.brier_baseline > 0
    # has_lift on the small synthetic universe is data-dependent; pin
    # the diagnostics + the boolean type, not the truth value (a 5-symbol
    # backtest can legitimately produce either outcome).
    assert 0.0 <= cal.brier_test <= 1.0
    assert 0.0 <= cal.auc_test <= 1.0
    assert isinstance(cal.has_lift, bool)


def test_composite_engine_exposes_probability_when_lift(tmp_path):
    save = tmp_path / "cal.json"
    Calibrator(
        breakpoints=[20.0, 50.0, 80.0],
        values=[0.3, 0.5, 0.8],
        horizon_bars=10, n_train=400, n_test=200,
        base_rate=0.55, brier_train=0.20, brier_test=0.21,
        brier_baseline=0.247, ece_test=0.02, auc_test=0.62,
        reliability=[], has_lift=True, lift_note="ok",
    ).save(save)
    engine = CompositeEngine(calibrator_path=save)
    res = engine.combine(
        "X", technical_score=85.0,
        fundamental_score=70.0, news_score=60.0,
    )
    assert res.calibrated_probability is not None
    assert res.calibrated_probability > 0.5
    assert res.horizon_days == 10


def test_composite_engine_keeps_probability_none_without_lift(tmp_path):
    save = tmp_path / "cal.json"
    Calibrator(
        breakpoints=[20.0, 80.0], values=[0.4, 0.6],
        horizon_bars=10, n_train=400, n_test=200,
        base_rate=0.5, brier_train=0.25, brier_test=0.25,
        brier_baseline=0.25, ece_test=0.0, auc_test=0.5,
        reliability=[], has_lift=False, lift_note="random data",
    ).save(save)
    engine = CompositeEngine(calibrator_path=save)
    res = engine.combine("X", technical_score=85.0)
    assert res.calibrated_probability is None


def test_composite_engine_without_calibrator_keeps_none(tmp_path, monkeypatch):
    # cd into a temp dir so the default runs/calibrator.json path is empty
    monkeypatch.chdir(tmp_path)
    engine = CompositeEngine()
    res = engine.combine("X", technical_score=85.0)
    assert res.calibrated_probability is None
    assert "calibrator" in res.calibration_note.lower()
