"""Tests for the isotonic calibrator and reliability diagnostics."""
from __future__ import annotations

import json

import numpy as np

from backtest.calibration import (
    Calibrator,
    auc,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    isotonic_predict,
    pav_isotonic,
    reliability_diagram,
)


# ----- PAV -----------------------------------------------------------------


def test_pav_already_monotonic_keeps_every_point():
    bp, vs = pav_isotonic([1, 2, 3, 4], [0.1, 0.3, 0.7, 0.9])
    np.testing.assert_array_equal(bp, [1, 2, 3, 4])
    np.testing.assert_array_almost_equal(vs, [0.1, 0.3, 0.7, 0.9])


def test_pav_pools_violating_pairs():
    # 0.5 then 0.2 violates monotonicity - the two should pool to 0.35
    bp, vs = pav_isotonic([1, 2, 3], [0.5, 0.2, 0.9])
    assert len(bp) == 2
    np.testing.assert_array_almost_equal(vs, [0.35, 0.9])


def test_pav_output_is_non_decreasing_on_random_input():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 200)
    y = rng.uniform(0, 1, 200)
    _, vs = pav_isotonic(x, y)
    assert (np.diff(vs) >= -1e-9).all()


def test_pav_empty_input():
    bp, vs = pav_isotonic([], [])
    assert bp.size == 0 and vs.size == 0


def test_isotonic_predict_clamps_outside_breakpoints():
    bp, vs = pav_isotonic([10, 20, 30], [0.1, 0.5, 0.9])
    out = isotonic_predict([0, 15, 25, 100], bp, vs)
    # below 10  -> first block, above 30 -> last block
    assert out[0] == 0.1
    assert out[-1] == 0.9


# ----- diagnostics ---------------------------------------------------------


def test_brier_score_known_values():
    # all predictions perfect -> Brier 0
    assert brier_score([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]) == 0.0
    # all predictions inverted -> Brier 1
    assert brier_score([1.0, 0.0, 1.0, 0.0], [0, 1, 0, 1]) == 1.0


def test_ece_perfect_predictions_is_zero():
    assert expected_calibration_error([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]) == 0.0


def test_auc_perfect_separator():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auc_random_predictions_is_about_half():
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 500)
    y = rng.integers(0, 2, 500)
    assert abs(auc(p, y) - 0.5) < 0.06


def test_auc_handles_all_one_class():
    assert auc([0.1, 0.5, 0.9], [1, 1, 1]) == 0.5


def test_reliability_diagram_buckets():
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(int)
    bins = reliability_diagram(p, y, n_bins=4)
    assert len(bins) == 4
    assert all(b["count"] > 0 for b in bins)


# ----- fit_calibrator ------------------------------------------------------


def test_fit_calibrator_detects_lift_on_separable_data():
    rng = np.random.default_rng(123)
    # synthetic: prob increases monotonically with score; labels drawn from prob
    train_scores = rng.uniform(0, 100, 500)
    train_probs = train_scores / 100.0
    train_labels = (rng.uniform(size=500) < train_probs).astype(int)
    test_scores = rng.uniform(0, 100, 300)
    test_probs = test_scores / 100.0
    test_labels = (rng.uniform(size=300) < test_probs).astype(int)

    cal = fit_calibrator(train_scores, train_labels,
                         test_scores, test_labels, horizon_bars=10)
    assert cal.has_lift is True
    assert cal.auc_test > 0.7
    assert cal.brier_test < cal.brier_baseline


def test_fit_calibrator_reports_no_lift_on_noise():
    rng = np.random.default_rng(7)
    scores = rng.uniform(0, 100, 600)
    labels = rng.integers(0, 2, 600)   # pure noise, no relationship
    cal = fit_calibrator(scores[:400], labels[:400],
                         scores[400:], labels[400:], horizon_bars=10)
    assert cal.has_lift is False
    assert "no predictive lift" in cal.lift_note


def test_calibrator_predict_returns_none_without_lift():
    cal = Calibrator(
        breakpoints=[10, 50, 90], values=[0.3, 0.5, 0.7],
        horizon_bars=10, n_train=100, n_test=50,
        base_rate=0.5, brier_train=0.25, brier_test=0.25,
        brier_baseline=0.25, ece_test=0.0, auc_test=0.5,
        reliability=[], has_lift=False,
    )
    assert cal.predict(60.0) is None


def test_calibrator_predict_when_has_lift():
    cal = Calibrator(
        breakpoints=[10, 50, 90], values=[0.3, 0.5, 0.7],
        horizon_bars=10, n_train=100, n_test=50,
        base_rate=0.5, brier_train=0.18, brier_test=0.19,
        brier_baseline=0.25, ece_test=0.02, auc_test=0.7,
        reliability=[], has_lift=True,
    )
    assert cal.predict(60.0) == 0.7


def test_calibrator_save_load_roundtrip(tmp_path):
    cal = Calibrator(
        breakpoints=[10, 50, 90], values=[0.3, 0.5, 0.7],
        horizon_bars=10, n_train=100, n_test=50,
        base_rate=0.5, brier_train=0.18, brier_test=0.19,
        brier_baseline=0.25, ece_test=0.02, auc_test=0.7,
        reliability=[{"bin_lo": 0.0, "bin_hi": 0.5, "predicted": 0.3,
                      "observed": 0.3, "count": 10}],
        has_lift=True, lift_note="ok",
    )
    path = tmp_path / "cal.json"
    cal.save(path)
    on_disk = json.load(open(path))
    assert on_disk["has_lift"] is True
    loaded = Calibrator.load(path)
    assert loaded.has_lift is True
    assert loaded.values == [0.3, 0.5, 0.7]
    assert loaded.predict(60.0) == 0.7
