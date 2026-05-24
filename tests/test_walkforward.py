"""Tests for rolling-window walk-forward calibration."""
from __future__ import annotations

import numpy as np

from backtest.events import PredictionPoint
from backtest.walkforward import run_walk_forward


def _make_points(n: int, seed: int = 0, signal: float = 0.0) -> list[PredictionPoint]:
    """Synthetic prediction points; `signal` ramps lift from 0 (noise) to 1
    (composite score is a near-perfect probability)."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        score = float(rng.uniform(0, 100))
        prob = signal * (score / 100.0) + (1.0 - signal) * 0.5
        label = int(rng.uniform() < prob)
        pts.append(PredictionPoint(
            symbol=f"S{i % 5}", date=f"2024-{1 + (i // 30) % 12:02d}-01",
            composite_score=score, technical_score=score,
            fundamental_score=None, sentiment_score=None,
            forward_return=(prob - 0.5) * 0.05, label=label,
        ))
    return pts


def test_walk_forward_produces_expected_fold_count(tmp_path):
    pts = _make_points(2000, signal=0.0)
    report = run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=200,
        save_path=tmp_path / "cal.json", horizon_bars=10,
    )
    # i starts at 500, advances by step=200, exit when i+200 > 2000
    # -> i in {500, 700, 900, 1100, 1300, 1500, 1700} -> 7 folds
    assert report.fold_count == 7
    assert report.n_points == 2000
    assert all(f.n_train == 500 and f.n_test == 200 for f in report.folds)


def test_walk_forward_aggregates_metrics(tmp_path):
    pts = _make_points(2000, signal=0.0)
    report = run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=300,
        save_path=tmp_path / "cal.json",
    )
    assert report.avg_brier_test > 0
    assert report.avg_brier_baseline > 0
    assert 0.0 <= report.avg_auc_test <= 1.0


def test_walk_forward_detects_stable_lift_on_signal(tmp_path):
    # strong synthetic signal -> every fold should show lift -> stable
    pts = _make_points(2000, signal=0.95)
    report = run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=300,
        save_path=tmp_path / "cal.json",
    )
    assert report.stable_lift is True
    assert report.lift_folds >= report.fold_count - 1
    assert report.final_calibrator.has_lift is True


def test_walk_forward_keeps_lift_off_on_noise(tmp_path):
    pts = _make_points(2000, signal=0.0)
    report = run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=300,
        save_path=tmp_path / "cal.json",
    )
    # noise -> few folds (if any) cross the threshold -> not stable
    assert report.stable_lift is False
    assert report.final_calibrator.has_lift is False


def test_walk_forward_raises_when_not_enough_points():
    pts = _make_points(200)
    try:
        run_walk_forward(
            settings=None, points=pts,
            train_size=500, test_size=200, step=100, save_path=None,
        )
    except RuntimeError as exc:
        assert "200" in str(exc) and "700" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_walk_forward_persists_final_calibrator(tmp_path):
    pts = _make_points(2000, signal=0.0)
    out = tmp_path / "cal.json"
    run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=300, save_path=out,
    )
    assert out.exists()


def test_walk_forward_threshold_is_strict_60_percent():
    """Regression: `max(1, int(round(folds * 0.6)))` was lax. With round(),
    4 folds needed only 2/4 (50%). math.ceil() gives the correct 3/5 for
    5 folds and 3/4 for 4."""
    import math
    from backtest.walkforward import _MIN_FOLDS_FOR_STABILITY, _STABLE_LIFT_FRACTION
    assert _STABLE_LIFT_FRACTION == 0.6
    assert _MIN_FOLDS_FOR_STABILITY >= 3
    assert math.ceil(5 * _STABLE_LIFT_FRACTION) == 3
    assert math.ceil(4 * _STABLE_LIFT_FRACTION) == 3
    assert math.ceil(3 * _STABLE_LIFT_FRACTION) == 2


def test_walk_forward_single_fold_cannot_declare_stable_lift(tmp_path):
    """Regression: round(1 * 0.6) = 1 and max(1, ...) = 1, so a single
    fold passing used to mark the calibrator stable. With the new
    _MIN_FOLDS_FOR_STABILITY guard, that's impossible."""
    pts = _make_points(800, seed=1, signal=0.95)
    report = run_walk_forward(
        settings=None, points=pts,
        train_size=500, test_size=200, step=300,
        save_path=tmp_path / "cal.json",
    )
    assert report.fold_count <= 2
    assert report.stable_lift is False
    assert report.final_calibrator.has_lift is False
