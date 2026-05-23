"""Rolling-window walk-forward calibration.

The single 70/30 split in harness.run_backtest is the simplest honest
out-of-sample test, but it cannot tell you whether the calibrator's
lift is stable through time or just a lucky split. Walk-forward rolls
a train+test window across the full point series and reports per-fold
diagnostics plus an aggregate. The final persisted calibrator is fit
on the most recent training window, and its `has_lift` flag is only
set when a clear majority of folds individually show lift - so the
calibrated probability the UI exposes reflects stability, not a single
fortunate slice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from backtest.calibration import (
    DEFAULT_CALIBRATOR_PATH,
    Calibrator,
    fit_calibrator,
)
from backtest.events import PredictionPoint, generate_points
from monitoring.logging import get_logger

log = get_logger("backtest.walkforward")

# 60% of folds must individually show lift before we consider the
# overall calibrator stable enough to expose probabilities.
_STABLE_LIFT_FRACTION = 0.6


@dataclass
class WalkForwardFold:
    fold: int
    n_train: int
    n_test: int
    brier_train: float
    brier_test: float
    brier_baseline: float
    auc_test: float
    ece_test: float
    has_lift: bool


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold] = field(default_factory=list)
    final_calibrator: Calibrator | None = None
    avg_brier_test: float = 0.0
    avg_brier_baseline: float = 0.0
    avg_auc_test: float = 0.0
    avg_ece_test: float = 0.0
    lift_folds: int = 0
    fold_count: int = 0
    stable_lift: bool = False
    horizon_bars: int = 0
    n_points: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_calibrator"] = (
            self.final_calibrator.to_dict() if self.final_calibrator else None
        )
        return d


def run_walk_forward(
    settings,
    symbols=None,
    horizon_bars: int = 10,
    train_size: int = 1000,
    test_size: int = 200,
    step: int = 200,
    save_path: str | Path | None = DEFAULT_CALIBRATOR_PATH,
    points: list[PredictionPoint] | None = None,
) -> WalkForwardReport:
    """Roll a (train, test) window through the point series and aggregate
    per-fold Brier / AUC / ECE. Persists the most-recent-window calibrator,
    gated on walk-forward stability rather than a single fold."""
    if step <= 0:
        raise ValueError("step must be positive")
    points = points or generate_points(
        settings, symbols=symbols, horizon_bars=horizon_bars,
    )
    if len(points) < train_size + test_size:
        raise RuntimeError(
            f"only {len(points)} prediction points; "
            f"need >= {train_size + test_size} for one walk-forward fold"
        )

    folds: list[WalkForwardFold] = []
    fold_idx = 0
    i = train_size
    while i + test_size <= len(points):
        train = points[i - train_size: i]
        test = points[i: i + test_size]
        cal = _fit_window(train, test, horizon_bars)
        folds.append(WalkForwardFold(
            fold=fold_idx,
            n_train=len(train), n_test=len(test),
            brier_train=cal.brier_train,
            brier_test=cal.brier_test,
            brier_baseline=cal.brier_baseline,
            auc_test=cal.auc_test,
            ece_test=cal.ece_test,
            has_lift=cal.has_lift,
        ))
        i += step
        fold_idx += 1

    if not folds:
        raise RuntimeError("no walk-forward folds produced - check window sizes")

    # final calibrator: fit on the most recent train_size window
    final_train = points[-(train_size + test_size): -test_size]
    final_test = points[-test_size:]
    final = _fit_window(final_train, final_test, horizon_bars)

    # walk-forward stability gate
    lift_folds = sum(1 for f in folds if f.has_lift)
    stable_lift = lift_folds >= max(1, int(round(len(folds) * _STABLE_LIFT_FRACTION)))

    if final.has_lift and not stable_lift:
        # the final window passed on its own, but walk-forward is shaky
        final.has_lift = False
        final.lift_note = (
            f"final window beat baseline but walk-forward is unstable "
            f"({lift_folds}/{len(folds)} folds showed lift) - probability "
            f"suppressed pending more stable history"
        )
    elif stable_lift and final.has_lift:
        final.lift_note = (
            f"walk-forward stable: {lift_folds}/{len(folds)} folds showed "
            f"lift; final window Brier {final.brier_test:.4f} vs baseline "
            f"{final.brier_baseline:.4f}"
        )

    if save_path is not None:
        final.save(save_path)
        log.info("saved walk-forward calibrator to %s", save_path)

    report = WalkForwardReport(
        folds=folds, final_calibrator=final,
        avg_brier_test=float(np.mean([f.brier_test for f in folds])),
        avg_brier_baseline=float(np.mean([f.brier_baseline for f in folds])),
        avg_auc_test=float(np.mean([f.auc_test for f in folds])),
        avg_ece_test=float(np.mean([f.ece_test for f in folds])),
        lift_folds=lift_folds, fold_count=len(folds),
        stable_lift=stable_lift,
        horizon_bars=horizon_bars, n_points=len(points),
    )
    log.info(
        "walk-forward: %d folds, %d/%d showed lift, avg Brier %.4f vs %.4f, "
        "avg AUC %.3f, stable=%s",
        report.fold_count, lift_folds, report.fold_count,
        report.avg_brier_test, report.avg_brier_baseline,
        report.avg_auc_test, stable_lift,
    )
    return report


def _fit_window(train, test, horizon_bars: int) -> Calibrator:
    train_scores = np.array([p.composite_score for p in train])
    train_labels = np.array([p.label for p in train])
    test_scores = np.array([p.composite_score for p in test])
    test_labels = np.array([p.label for p in test])
    return fit_calibrator(
        train_scores, train_labels, test_scores, test_labels,
        horizon_bars=horizon_bars,
    )
